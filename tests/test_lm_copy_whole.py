#!/usr/bin/env python3
"""MOE_LM_COPY_WHOLE (stream_fast._LM_COPY_WHOLE): a layer-major layer read whole
copies the store's residents slot to slot instead of reading them again. On test_union_lm's toy and passes
(MOE_READPATH=whole as the reference: every layer whole, nothing copied), whole with the flag and auto with
the flag give bit-equal logits and cache arrays at cache bias 0 and 0.5, copy rows (lm_copied > 0) and read
fewer bytes; a copy from another slot is caught. Metal and CPU take the host copy (_LMReader._read); CUDA's
copy runs in cuda_stage.LMStage.take, which this file reaches only when run on CUDA.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_lm_copy_whole.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_union_lm as tul
import stream_fast


def arm(out, mode, bias):
    stream_fast._LM_COPY_WHOLE = True
    try:
        return tul.run(out, mode, bias)
    finally:
        stream_fast._LM_COPY_WHOLE = False


def main():
    print(f"   backend {stream_fast.BACKEND}, CUDA stage {'on' if stream_fast._CUDA_STAGE_ON else 'off'}")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "q3n"
        tul.tlp.build(out)
        for bias in (0.0, 0.5):
            # As test_union_lm: at bias 0.5 the put-back follows the regions written, so it is off there.
            stream_fast._KEEP = bias == 0.0
            ref = tul.run(out, "whole", bias)
            assert ref[2]["lm_copied"] == 0, ref[2]
            for mode in ("whole", "auto"):
                got = arm(out, mode, bias)
                tul.check(ref, got, f"copy-whole {mode}, bias {bias}")
                st = got[2]
                assert st["lm_copied"] > 0 and st["lm_read_b"] < ref[2]["lm_read_b"], (st, ref[2])
                if mode == "whole":
                    assert st["lm_union"] == 0, st
                print(f"   bias {bias}, copy-whole {mode}: {len(got[0])} passes' logits, {len(got[1])} cache "
                      f"arrays == plain whole; {st['lm_whole']} whole / {st['lm_union']} union layers, "
                      f"{st['lm_copied']} rows copied, {st['lm_read_b'] / 1e6:.2f} MB read "
                      f"(plain whole {ref[2]['lm_read_b'] / 1e6:.2f} MB)")
        stream_fast._KEEP = True
        ref = tul.run(out, "whole")
        real = stream_fast._LMReader._resident

        def wrong_slot(self, glu, e):
            sl = real(self, glu, e)
            others = [s for s in self.res.get(id(glu), {}).values() if s != sl]
            return others[0] if sl is not None and others else sl
        stream_fast._LMReader._resident = wrong_slot
        try:
            got = arm(out, "whole", 0.0)
        finally:
            stream_fast._LMReader._resident = real
        assert not (tul.tlp.same(got[0], ref[0]) and tul.tlp.same(got[1], ref[1])), "mutant not caught"
        print("   mutant caught: a whole layer's resident copied from another slot")
    print("test_lm_copy_whole: all passed")


if __name__ == "__main__":
    main()
