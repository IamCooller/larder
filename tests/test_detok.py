#!/usr/bin/env python3
"""The Rust-backed streaming detokenizer (stream_fast._lean_detokenizer) emits
exactly mlx_lm's BPE streaming text, segment by segment. Tokenizer files only,
no weights; skipped when no local Qwen tokenizer is present."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import support  # noqa: E402

CANDIDATES = [
    *[p for p in [support.tokenizer_dir()] if p],
    Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
    Path("<a Qwen3 checkpoint directory>"),
]
TEXT = ("Grüße, Welt! 你好, Hello, naïve café 🙂\n```python\ndef f(x):\n    return x ** 2\n```\n"
        "<tool_call>\n<function=weather>\n<parameter=place>\nBerlin\n</parameter>\n</function>\n</tool_call>")


def stream(detok, ids):
    detok.reset()
    out = []
    for t in ids:
        detok.add_token(t)
        out.append(detok.last_segment)
    detok.finalize()
    out.append(detok.last_segment)
    return out


def main():
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer
    from mlx_lm.utils import load_tokenizer
    import stream_fast

    paths = [p for p in CANDIDATES if (p / "tokenizer.json").exists()]
    if not paths:
        print("test_detok: skipped (no local Qwen tokenizer)")
        return
    for path in paths:
        tok = load_tokenizer(path)
        stock = tok.detokenizer
        if type(stock) is not BPEStreamingDetokenizer:
            print(f"  {path.name}: not a BPE streaming detokenizer, untouched")
            continue
        lean = stream_fast._lean_detokenizer(tok).detokenizer
        assert type(lean) is not BPEStreamingDetokenizer
        ids = tok.encode(TEXT, add_special_tokens=False)
        ids += [tok.eos_token_ids and sorted(tok.eos_token_ids)[0] or 0]
        a, b = stream(stock, ids), stream(lean, ids)
        assert a == b, next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        n = len(lean.tokenmap)
        assert n == len(stock.tokenmap)
        for i in range(0, n, max(1, n // 5000)):
            assert lean.tokenmap[i] == stock.tokenmap[i], i
        print(f"  {path.name}: {len(ids)} tokens streamed identically, "
              f"{n} ids sampled every {max(1, n // 5000)} match")
    print("test_detok: ok")


if __name__ == "__main__":
    main()
