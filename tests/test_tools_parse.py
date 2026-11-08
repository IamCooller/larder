#!/usr/bin/env python3
"""tools.parse on the replies a coding agent really breaks on: calls cut short, bare function blocks, values quoting the call's own tags.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_tools_parse.py

No model: the tool list has a coding agent's Bash, Edit and Read parameters, and each case is a reply text with the calls and the left-over text it must give.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools as toolfmt                                            # noqa: E402

S = {"type": "string"}


def tool(name, props, required):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": props, "required": required}}}


TOOLS = [
    tool("Bash", {"command": S, "timeout": {"type": "number"}, "description": S,
                  "run_in_background": {"type": "boolean"}}, ["command"]),
    tool("Edit", {"file_path": S, "old_string": S, "new_string": S, "replace_all": {"type": "boolean"}},
         ["file_path", "old_string", "new_string"]),
    tool("Read", {"file_path": S, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["file_path"]),
]
E = "/w/calc.py"


def call(name, **params):
    ps = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{ps}</function>\n</tool_call>"


LS = ("Bash", {"command": "ls"})
EDIT_TAGS = "    <parameter=city>\n    Berlin\n    </parameter>\n    </function>\n    </tool_call>"
HEAD = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n"

# name: (reply, calls, text left over)
CASES = {
    "whole call": (call("Bash", command="ls"), [LS], ""),
    "text after the call": ("I will look.\n" + call("Bash", command="ls") + "\nThen I will fix it.", [LS],
                            "I will look.\n\nThen I will fix it."),
    "two calls": (call("Read", file_path=E) + "\n" + call("Bash", command="ls -la"),
                  [("Read", {"file_path": E}), ("Bash", {"command": "ls -la"})], ""),
    "inline params": ("<tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>", [LS], ""),
    "JSON form": ('<tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call>', [LS], ""),
    "types": (call("Bash", command="ls", timeout="120000") + call("Read", file_path=E, offset="10"),
              [("Bash", {"command": "ls", "timeout": 120000.0}), ("Read", {"file_path": E, "offset": 10})], ""),
    "indent and trailing newline kept": (call("Edit", file_path=E, old_string="    return a - b\n", new_string="    return a + b\n"),
                                         [("Edit", {"file_path": E, "old_string": "    return a - b\n", "new_string": "    return a + b\n"})], ""),
    "empty new_string": (call("Edit", file_path=E, old_string="x = 1", new_string=""),
                         [("Edit", {"file_path": E, "old_string": "x = 1", "new_string": ""})], ""),
    "call in a markdown fence": ("Example:\n```\n" + call("Bash", command="rm -rf build") + "\n```",
                                 [("Bash", {"command": "rm -rf build"})], "Example:\n```\n\n```"),
    # Cut short at the end of the reply or by a missing closing tag.
    "no </function>": (HEAD + "</tool_call>", [LS], ""),
    "no </tool_call>": (HEAD + "</function>", [LS], ""),
    "no </function>, no </tool_call>": (HEAD.rstrip(), [LS], ""),
    "text, then a call with no </tool_call>": ("Looking.\n" + HEAD + "</function>\n", [LS], "Looking."),
    "no </tool_call> between two calls": (HEAD + "</function>\n" + call("Read", file_path=E),
                                          [LS, ("Read", {"file_path": E})], ""),
    "second call unclosed": (call("Read", file_path=E) + "\n" + HEAD + "</function>",
                             [("Read", {"file_path": E}), LS], ""),
    "last param lost its </parameter>": (
        "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/w/a.py\n</parameter>\n<parameter=old_string>\nx\n</parameter>\n"
        "<parameter=new_string>\ny\n</function>\n</tool_call>",
        [("Edit", {"file_path": "/w/a.py", "old_string": "x", "new_string": "y"})], ""),
    "bare <function=> block": ("<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>", [LS], ""),
    "text, then a bare block": ("Checking.\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>", [LS], "Checking."),
    # What must not turn into a call.
    "cut inside a value": ("<tool_call>\n<function=Bash>\n<parameter=command>\nls -", [], "<tool_call>\n<function=Bash>\n<parameter=command>\nls -"),
    "cut with a required parameter missing": (
        "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/w/a.py\n</parameter>\n<parameter=old_string>\nx\n</parameter>", [],
        "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/w/a.py\n</parameter>\n<parameter=old_string>\nx\n</parameter>"),
    "bare block of a tool that is not listed": ("<function=rm>\n<parameter=path>\n/\n</parameter>\n</function>", [],
                                                "<function=rm>\n<parameter=path>\n/\n</parameter>\n</function>"),
    "bare block that never closes": ("<function=Bash>\n<parameter=command>\nls\n</parameter>", [],
                                     "<function=Bash>\n<parameter=command>\nls\n</parameter>"),
    "tag named in prose": ("Write <function=NAME> then the parameters.", [], "Write <function=NAME> then the parameters."),
    "block that is no call": ("<tool_call>\nnot a call\n</tool_call>", [], "<tool_call>\nnot a call\n</tool_call>"),
    "function_calls XML": ('<function_calls>\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n</invoke>\n</function_calls>', [],
                      '<function_calls>\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n</invoke>\n</function_calls>'),
    # A value quoting the call's own tags: the declared parameter names and the required ones say where it ends.
    "value quotes </function></tool_call> (an edit of tools.py's docstring)": (
        call("Edit", file_path="/w/tools.py", old_string=EDIT_TAGS, new_string="    </tool_call>\n"),
        [("Edit", {"file_path": "/w/tools.py", "old_string": EDIT_TAGS, "new_string": "    </tool_call>\n"})], ""),
    "value quotes </parameter> then <parameter=y>": (
        call("Edit", file_path="/w/t.txt", old_string="x\n</parameter>\n<parameter=y>\nz", new_string="q"),
        [("Edit", {"file_path": "/w/t.txt", "old_string": "x\n</parameter>\n<parameter=y>\nz", "new_string": "q"})], ""),
    "value quotes two whole parameters": (
        call("Edit", file_path="/w/t.py", old_string="<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>", new_string="x"),
        [("Edit", {"file_path": "/w/t.py", "old_string": "<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>", "new_string": "x"})], ""),
    "value quotes </parameter>\\n": (
        call("Edit", file_path="/w/t.py", old_string="a = '</parameter>\\n'\nb = 1", new_string="a = 2"),
        [("Edit", {"file_path": "/w/t.py", "old_string": "a = '</parameter>\\n'\nb = 1", "new_string": "a = 2"})], ""),
    "Bash value quotes the whole tail": (
        call("Bash", command="echo '</parameter>\n</function>\n</tool_call>'"),
        [("Bash", {"command": "echo '</parameter>\n</function>\n</tool_call>'"})], ""),
    "unicode": (call("Edit", file_path=E, old_string="s = 'Zürich — 東京 🚀'", new_string="s = 'ok'"),
                [("Edit", {"file_path": E, "old_string": "s = 'Zürich — 東京 🚀'", "new_string": "s = 'ok'"})], ""),
    # A parameter the tool does not list, as real replies carry them: after a number or a name it does not list it still ends the value.
    "extra parameter after a number": (
        call("Read", file_path=E, offset="420", limit="18", timeout="30000"),
        [("Read", {"file_path": E, "offset": 420, "limit": 18, "timeout": "30000"})], ""),
    "run of extra parameters": (
        call("Read", file_path=E, limit="15", offset="448", limit2="14", offset2="446", limit3="10"),
        [("Read", {"file_path": E, "limit": 15, "offset": 448, "limit2": "14", "offset2": "446", "limit3": "10"})], ""),
    # After a text value too, once every required parameter is in (grep's pattern once swallowed an extra path).
    "extra parameter after a text value": (call("Bash", command="ls", path="/w"), [("Bash", {"command": "ls", "path": "/w"})], ""),
    "two extra parameters after a text value": (
        call("Bash", command="grep -n x", pattern_2="y", path="/w"),
        [("Bash", {"command": "grep -n x", "pattern_2": "y", "path": "/w"})], ""),
    "extra parameter, then a second call": (
        call("Bash", command="ls", path="/w") + "\n" + call("Read", file_path=E),
        [("Bash", {"command": "ls", "path": "/w"}), ("Read", {"file_path": E})], ""),
    "last value quotes a whole call": (
        call("Edit", file_path="/w/tools.py", old_string="x",
             new_string="    <parameter=a>\n    1\n    </parameter>\n    <parameter=b>\n    2\n    </parameter>\n    </function>\n    </tool_call>"),
        [("Edit", {"file_path": "/w/tools.py", "old_string": "x",
                   "new_string": "    <parameter=a>\n    1\n    </parameter>\n    <parameter=b>\n    2\n    </parameter>\n    </function>\n    </tool_call>"})], ""),
    # A required parameter the model left out: still one (partial) call, so the client can say what is missing.
    "required parameter left out": (
        "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/w/a.py\n</parameter>\n<parameter=old_string>\nx\n</parameter>\n</function>\n</tool_call>",
        [("Edit", {"file_path": "/w/a.py", "old_string": "x"})], ""),
}


def got(reply, tools=TOOLS):
    text, calls = toolfmt.parse(reply, tools)
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls], text


def main():
    bad = []
    for name, (reply, calls, text) in CASES.items():
        have = got(reply)
        if have != (calls, text):
            bad.append(name)
            print(f"FAIL {name}\n  want {(calls, text)!r}\n  have {have!r}")
    assert not bad, bad
    # Without a tool list nothing is declared: any parameter name ends a value, and a bare block is not a call.
    assert got(call("Bash", command="ls"), None) == ([LS], "")
    assert got("<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>", None)[0] == []
    # A reply of coder_base's runs as the model wrote it, against the agent bench's grep tool.
    grep = [tool("grep", {"pattern": S}, ["pattern"])]
    real = "<tool_call>\n<function=grep>\n<parameter=pattern>\nresp\n</parameter>\n<parameter=path>\nmoe_stream.py\n</parameter>\n</function>\n</tool_call>"
    assert got(real, grep) == ([("grep", {"pattern": "resp", "path": "moe_stream.py"})], ""), got(real, grep)
    print(f"tools.parse: {len(CASES)} cases hold")


if __name__ == "__main__":
    main()
