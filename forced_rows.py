#!/usr/bin/env python3
"""Grammar-forced verify rows: a Forcer walks an xgrammar matcher (or the pure-Python `_Lite` walker where xgrammar is not installed) over the committed tokens and only PROPOSES the canonical tokenization of what a tool schema or JSON response_format leaves no choice about, so the target still checks every row and the reply stays its greedy argmax."""

import copy
import json
import sys
from functools import lru_cache
from engine_log import note

_notice = [False]


@lru_cache(maxsize=4)
def _compiler(hf_tokenizer):
    import xgrammar as xgr
    return xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(hf_tokenizer))


@lru_cache(maxsize=16)
def _tool_grammar(hf_tokenizer, tools_json):
    """One grammar over every tool of the request: anyOf the per-tool
    {"name": const, "arguments": schema} objects, framed by the newlines
    Qwen3's template puts inside the call tags. Before the name is decided the
    shared prefix is forced; after, the one tool's own keys are."""
    import xgrammar as xgr
    calls = []
    for t in json.loads(tools_json):
        fn = t.get("function", t)
        calls.append({"type": "object",
                      "properties": {"name": {"const": fn["name"]},
                                     "arguments": fn.get("parameters", {})},
                      "required": ["name", "arguments"],
                      "additionalProperties": False})
    body = xgr.Grammar.from_json_schema(
        json.dumps({"anyOf": calls}), any_whitespace=False,
        separators=(", ", ": "))
    nl = xgr.Grammar.from_ebnf('root ::= "\\n"')
    return _compiler(hf_tokenizer).compile_grammar(
        xgr.Grammar.concat(nl, body, nl))


@lru_cache(maxsize=16)
def _xml_grammar(hf_tokenizer, tools_json):
    """One grammar over every tool of the request in the qwen3_coder layout
    (the text between the call tags, as the chat template writes it):
    parameters in declared order, optional ones skippable, each value
    anything up to the first `\\n</parameter>` -- the parser's own rule, so
    a value may span lines."""
    lit = lambda s: json.dumps(s, ensure_ascii=False)   # noqa: E731
    calls = []
    for t in json.loads(tools_json):
        fn = t.get("function", t)
        params = fn.get("parameters") or {}
        need = set(params.get("required", ()))
        call = [lit("\n<function=" + fn["name"] + ">\n")]
        for key in params.get("properties", {}):
            p = lit("<parameter=" + key + ">\n") + " value " + lit("\n</parameter>\n")
            call.append(p if key in need else f"({p})?")
        calls.append("(" + " ".join(call + [lit("</function>\n")]) + ")")
    return _compiler(hf_tokenizer).compile_grammar(
        "root ::= " + " | ".join(calls) + "\n"
        'value ::= TagDispatch(loop_after_dispatch=false, excludes=("\\n</parameter>"))\n')


@lru_cache(maxsize=16)
def _reply_grammar(hf_tokenizer, schema_json):
    """The whole reply is one JSON value of this schema, laid out like the
    tool grammar's arguments."""
    return _compiler(hf_tokenizer).compile_json_schema(
        schema_json, any_whitespace=False, separators=(", ", ": "))


class Forcer:
    """feed() every committed token, propose() what the grammar forces next.
    fork() is a throwaway copy a round can walk over its own draft tokens.

    With an `open_id` the grammar covers the span that token opens and
    `close_id` ends (a tool call); without one it covers the reply from its
    first token, and `close_id` is the end of turn (response_format).
    `grammar` is xgrammar's compiled grammar or a `_Lite` one."""

    def __init__(self, grammar, encode, open_id, close_id):
        if isinstance(grammar, _Lite):
            self._matcher = grammar.matcher
        else:
            import xgrammar as xgr
            self._matcher = lambda: xgr.GrammarMatcher(
                grammar, terminate_without_stop_token=True)
        self._encode = encode
        self.open_id, self.close_id = open_id, close_id
        # live matcher while inside the grammar's span, else None
        self._m = self._matcher() if open_id is None else None

    @property
    def active(self):
        return self._m is not None

    def fork(self):
        f = copy.copy(self)
        f._m = self._guard(self._m.fork) if self._m is not None else None
        return f

    def _guard(self, call, *args):
        # The built-in matcher is Python and new; a proposal only saves time,
        # so any error in it switches the forcer off for this call instead of
        # failing the request.
        try:
            return call(*args)
        except Exception as e:
            print(f"  forced rows: matcher error ({type(e).__name__}: {e}), "
                  f"off for this call", flush=True)
            return None

    def feed(self, tok):
        if self._m is None:
            if tok == self.open_id:
                self._m = self._guard(self._matcher)
        elif tok == self.close_id or not self._guard(self._m.accept_token, tok):
            # the call ended, or the model left the schema: off until the
            # next <tool_call> (a JSON reply has no next one), the output
            # stays the target's either way
            self._m = None

    def propose(self):
        ids = self._guard(self._propose)
        if ids is None:             # the matcher failed: off, see _guard
            self._m = None
            return []
        return ids

    def _propose(self):
        m = self._m
        if m is None:
            return []
        if m.is_terminated():
            return [self.close_id]
        s = m.find_jump_forward_string()
        if not s:
            return []
        ids = self._encode(s, add_special_tokens=False)
        # Walk the ids to see whether they end the call, then undo the walk:
        # only then is the last id safe (nothing but </tool_call>, or the
        # end of turn, follows).
        ok = 0
        while ok < len(ids) and m.accept_token(ids[ok]):
            ok += 1
        done = ok == len(ids) and m.is_terminated()
        if ok:
            m.rollback(ok)
        return ids + [self.close_id] if done else ids[:min(ok, len(ids) - 1)]


# --- the same grammars without xgrammar -------------------------------------
#
# xgrammar costs the process torch (~150 MB) plus its own TokenizerInfo and
# GrammarCompiler (~90 MB), which a 2 GB client cannot spare, so the same
# grammars, small and regular (a call's scaffolding, the schema's names in
# declared order, free text in between), are rebuilt here from the same
# schemas as items that a matcher walks byte by byte as a set of live parses
# ("threads"), the jump-forward string being the bytes every thread agrees on
# next, up to a choice, free text or the end. Items:
#   bytes                  to match
#   _Alt                   one of its sequences (tuples of items)
#   ("str", soft, esc)     a JSON string's body, up to its closing quote
#   ("num", soft, started) a JSON number (loose: its characters)
#   ("until", soft, stop, k)  any text up to and including the first `stop`,
#                          k bytes of it matched so far
# A soft leaf is free text behind a value the schema predicts (an enum's, a
# number's single line): proposals follow the prediction, and a model that
# writes something else only loses them until the value ends -- the call
# parses on, where xgrammar's JSON grammar would switch off.

class _Alt:
    """One of several item sequences. Compared by identity: grammars share
    these nodes and a repeat is a cycle, so a thread hashes in its length."""
    __slots__ = ("seqs",)

    def __init__(self, *seqs):
        self.seqs = seqs


def _star(seq, then):
    """`seq` any number of times, then the sequence `then`."""
    loop = _Alt()
    loop.seqs = (then, seq + (loop,))
    return loop


_STR = ("str", False, False)
_VALUE = _Alt()                                  # any JSON value
_OBJECT = (b"{", _Alt((b"}",), (b'"', _STR, b": ", _VALUE, _star(
    (b', "', _STR, b": ", _VALUE), (b"}",)))))
_ARRAY = (b"[", _Alt((b"]",), (_VALUE, _star((b", ", _VALUE), (b"]",)))))
_VALUE.seqs = ((b'"', _STR), (("num", False, False),), (b"true",),
               (b"false",), (b"null",), _OBJECT, _ARRAY)


def _ordered(parts, end, sep=b""):
    """(sequence, required) parts in declared order, an optional one
    skippable, then `end`, with `sep` before every part but the first one
    written: how xgrammar lays out an object's properties, and the
    qwen3_coder template a call's parameters."""
    after = {True: (end,), False: (end,)}       # keyed by "none written yet"
    for seq, need in reversed(parts):
        after = {first: (_Alt(((sep,) if sep and not first else ()) + seq
                              + after[False], *(() if need else (after[first],))),)
                 for first in (True, False)}
    return after[True]


def _json_value(s):
    """One JSON value of schema `s` in the tool grammar's layout (no
    whitespace, ", " and ": "), any value where the schema says less, as
    xgrammar does: {"type": "object"} is any object, "properties": {} only {},
    a declared object only its keys in order."""
    if not isinstance(s, dict):
        return (_VALUE,)
    vals = s["enum"] if isinstance(s.get("enum"), list) else (
        [s["const"]] if "const" in s else None)
    if vals:
        alts = [(json.dumps(v, ensure_ascii=False).encode(),) for v in vals]
        if all(isinstance(v, str) for v in vals):
            alts.append((b'"', ("str", True, False)))
        return (_Alt(*alts),)
    for k in ("anyOf", "oneOf"):
        if isinstance(s.get(k), list):
            return (_Alt(*map(_json_value, s[k])),)
    t = s.get("type")
    if isinstance(t, list):
        return (_Alt(*(_json_value({**s, "type": x}) for x in t)),)
    if t == "string":
        return (b'"', _STR)
    if t in ("integer", "number"):
        return (("num", False, False),)
    if t == "boolean":
        return (_Alt((b"true",), (b"false",)),)
    if t == "null":
        return (b"null",)
    if t == "object" and isinstance(s.get("properties"), dict):
        need = set(s.get("required", ()))
        return (b"{",) + _ordered(
            [((json.dumps(k, ensure_ascii=False).encode() + b": ",)
              + _json_value(v), k in need) for k, v in s["properties"].items()],
            b"}", b", ")
    if t == "array" and isinstance(s.get("items"), dict):
        item = _json_value(s["items"])
        return (b"[", _Alt((b"]",), item + (_star((b", ",) + item, (b"]",)),)))
    return {"object": _OBJECT, "array": _ARRAY}.get(t, (_VALUE,))


def _xml_value(s):
    """A parameter's value and the `\\n</parameter>` that ends it -- the
    first one does (the parser's rule), so a value may span lines. xgrammar's
    grammar stops there; this one also knows two values ahead: one from a
    string enum once its first letters pick it, and a number or boolean,
    which the parser reads as one line, so its newline is followed by
    </parameter>. Free text stays behind both as a soft fallback."""
    end = b"\n</parameter>"
    if not isinstance(s, dict):
        return (("until", False, end, 0),)
    soft = (("until", True, end, 0),)
    vals = s.get("enum")
    if isinstance(vals, list) and vals and all(isinstance(v, str) for v in vals):
        return (_Alt(*[(v.encode() + end,) for v in vals], soft),)
    if s.get("type") in ("integer", "number", "boolean"):
        return (_Alt((("until", False, b"\n", 0), b"</parameter>"), soft),)
    return (("until", False, end, 0),)


class _Lite:
    """A grammar as items (`root`, a sequence) and the tokenizer that reads
    ids as bytes: Qwen's vocabularies are byte-level BPE, and a letter can
    come in two tokens."""

    def __init__(self, root, hf):
        import mlx_lm.tokenizer_utils as tu
        bpe = tu.BPEStreamingDetokenizer
        # mlx_lm 0.32 moved the byte map from the class to a module function.
        if hasattr(tu, "_byte_decoder"):
            byte = tu._byte_decoder()
        else:
            bpe.make_byte_decoder()
            byte = bpe._byte_decoder
        self.root, self._hf, self._byte = root, hf, byte

    def matcher(self):
        return _LiteMatcher(self, frozenset(_expand(self.root)))

    def bytes_of(self, tok):
        s = self._hf.convert_ids_to_tokens(tok) or ""
        try:
            return bytes(self._byte[c] for c in s)
        except KeyError:            # an added token kept as plain text
            return s.encode()


@lru_cache(maxsize=16)
def _lite(hf, fmt, spec):
    """_tool_grammar/_xml_grammar/_reply_grammar's languages as items."""
    spec = json.loads(spec)
    if fmt == "reply":
        return _Lite(_json_value(spec), hf)
    calls = []
    for t in spec:
        fn = t.get("function", t)
        params = fn.get("parameters") or {}
        if fmt == "json":
            calls.append((json.dumps(fn["name"], ensure_ascii=False)[1:].encode()
                          + b', "arguments": ',) + _json_value(params) + (b"}",))
            continue
        need = set(params.get("required", ()))
        calls.append((fn["name"].encode() + b">\n",) + _ordered(
            [((f"<parameter={k}>\n".encode(),) + _xml_value(v) + (b"\n",), k in need)
             for k, v in (params.get("properties") or {}).items()],
            b"</function>\n"))
    return _Lite((b'\n{"name": "', _Alt(*calls), b"\n") if fmt == "json" else
                 (b"\n<function=", _Alt(*calls)), hf)


def _expand(t):
    """The threads sequence `t` stands for, each headed by an item that
    reads bytes, or empty where the grammar may end."""
    if not t:
        return [t]
    head, rest = t[0], t[1:]
    if isinstance(head, _Alt):
        return [x for seq in head.seqs for x in _expand(seq + rest)]
    if head == b"":
        return _expand(rest)
    if head == ("num", False, True):             # a number may end anywhere
        return [t] + _expand(rest)
    return [t]


def _step(t, b):
    """The threads `t` becomes after byte `b`."""
    if not t:
        return []
    head, rest = t[0], t[1:]
    if isinstance(head, bytes):
        return _expand((head[1:],) + rest) if head[0] == b else []
    if head[0] == "str":
        _, soft, esc = head
        if b == 0x22 and not esc:
            return _expand(rest)
        return [] if b < 0x20 else [(("str", soft, b == 0x5C and not esc),) + rest]
    if head[0] == "num":
        ok = b in (b"0123456789+-.eE" if head[2] else b"-0123456789")
        return _expand((("num", False, True),) + rest) if ok else []
    _, soft, stop, k = head                     # "until"
    seen = stop[:k] + bytes((b,))
    k = next(j for j in range(len(seen), -1, -1) if seen.endswith(stop[:j]))
    return _expand(rest) if k == len(stop) else [(("until", soft, stop, k),) + rest]


class _LiteMatcher:
    """The xgrammar GrammarMatcher calls Forcer makes, over a _Lite grammar:
    the live threads after each accepted token, kept for rollback."""

    def __init__(self, grammar, live):
        self._g, self._live = grammar, [live]

    def fork(self):
        return _LiteMatcher(self._g, self._live[-1])

    def accept_token(self, tok):
        live = self._live[-1]
        for b in self._g.bytes_of(tok):
            live = frozenset(x for t in live for x in _step(t, b))
            if not live:
                return False
        self._live.append(live)
        return True

    def rollback(self, n):
        del self._live[len(self._live) - n:]

    def is_terminated(self):
        return self._live[-1] == {()}

    def find_jump_forward_string(self):
        live, out = self._live[-1], bytearray()
        while True:
            firm = [t for t in live if not (t and isinstance(t[0], tuple) and t[0][1])]
            nxt = {t[0][0] if t and isinstance(t[0], bytes) else None for t in firm}
            if len(nxt) != 1 or None in nxt:
                break
            out.append(nxt.pop())
            live = frozenset(x for t in live for x in _step(t, out[-1]))
        # Text for the tokenizer: nothing if it would start inside a letter
        # (the last token ended mid-way through one), and a letter the
        # choice splits is left to the model.
        return "" if out and 0x80 <= out[0] < 0xC0 else out.decode("utf-8", "ignore")


def _xgrammar():
    # The lean-host torch shadow (server.py/chat.py) only stands in while
    # nothing needs it; xgrammar's own __init__ chain imports torch for real.
    # Lift the shadow before this import, the one call
    # every make() path makes first, and put it back if xgrammar itself is
    # the thing missing, so a process that never sees a tool request keeps
    # paying nothing for torch.
    shadowed = sys.modules.get("torch") is None and "torch" in sys.modules
    if shadowed:
        del sys.modules["torch"]
    try:
        import xgrammar  # noqa: F401
    except ImportError:
        # xgrammar's own __init__ chain can import torch for real before
        # hitting a *different* missing dependency further down (pydantic,
        # tvm_ffi, ...); that ImportError still lands here, but torch is by
        # then a real, fully-loaded module. Re-shadowing it in that case
        # doesn't undo the load -- torch's C extension stays resident -- so
        # the next lift just re-runs torch/__init__.py's registration code
        # over an already-registered extension and raises RuntimeError
        # ("Only a single TORCH_LIBRARY can be used to register the
        # namespace ..."), which isn't an ImportError and escapes make().
        # Only restore the shadow when torch itself never loaded.
        if shadowed and "torch" not in sys.modules:
            sys.modules["torch"] = None
        if not _notice[0]:
            _notice[0] = True
            note("  forced rows: xgrammar not installed, built-in grammars")
        return False
    return True


def _schema(response_format):
    """The JSON schema an OpenAI response_format asks for, or None: a
    json_schema one carries it (none given means any object), json_object
    means any object, "text" or anything else means no grammar."""
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "json_object":
        return {"type": "object"}
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        return (spec.get("schema") if isinstance(spec, dict) else None) \
            or {"type": "object"}
    return None


def make(tokenizer, tools, response_format=None):
    """A forcer for this request, or None.

    Tools come first when a request carries both: the reply may be a call.
    A tool forcer needs a call format it has a grammar for -- Qwen3's JSON
    (mlx_lm's json_tools parser) or Qwen3.5's XML (qwen3_coder) -- and call
    tags that are single special tokens. A response_format forcer needs a
    JSON response_format and an end-of-turn token. Without xgrammar the
    built-in grammars stand in (_Lite), and a schema that does not compile
    turns it off: the forcer only ever saves time, so it never fails a
    request.
    """
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    if tools:
        parser = getattr(getattr(tokenizer, "tool_parser", None), "__module__", "")
        fmt = ("json" if parser.endswith(".json_tools") else
               "xml" if parser.endswith(".qwen3_coder") else None)
        start = tuple(getattr(tokenizer, "tool_call_start_tokens", None) or ())
        end = tuple(getattr(tokenizer, "tool_call_end_tokens", None) or ())
        if fmt is None or len(start) != 1 or len(end) != 1:
            return None
        # Key order is kept: the schema grammar wants properties in declared
        # order, and sorting them made every call whose model writes them in
        # that order leave the grammar (the rejected rows of ab_forced.json).
        spec, open_id, close_id = json.dumps(tools), start[0], end[0]
    else:
        schema = _schema(response_format)
        eos = getattr(tokenizer, "eos_token_id", None)
        if schema is None or eos is None:
            return None
        fmt, spec, open_id, close_id = "reply", json.dumps(schema), None, eos
    xgr = _xgrammar()
    try:
        grammar = ({"json": _tool_grammar, "xml": _xml_grammar,
                    "reply": _reply_grammar}[fmt](hf, spec) if xgr else
                   _lite(hf, fmt, spec))
    except Exception as e:      # whatever a bad schema raises
        print(f"  forced rows: grammar not compiled ({e}), off")
        return None
    return Forcer(grammar, hf.encode, open_id, close_id)
