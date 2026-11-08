"""Quantise the downloaded MTP head to 4 bits and write it in mlx_lm key names.

Run as its own process: the bf16 head is 1.7 GB and the target model is another
1.75 GB resident, and this project has a hard 4 GB ceiling on process RSS.
Packing separately keeps each peak near 2 GB.

Two checkpoint conventions are handled here, both of which would otherwise be
silent wrong-weights bugs rather than crashes:

1. Norm weights are stored zero-centred. mlx_lm's qwen3_5 sanitize adds 1.0 to
   every RMSNorm weight when the checkpoint carries an MTP head, so the same
   shift has to be applied to the head's own norms.
2. The experts are fused as gate_up_proj [E, 2*inter, hidden] and down_proj
   [E, hidden, inter], which is already [E, out, in]; the split in
   qwen3_vl_moe.sanitize is for a transposed layout and must NOT be copied.
   Verified against the same repo's main-model layer 3, identical shapes.

Which half of gate_up is the gate is a convention this file cannot verify from
shapes alone, so --swap-gate-up writes the other order and an acceptance
measurement tells both apart.
"""
import argparse, json, sys
from pathlib import Path
import mlx.core as mx

GROUP = 64
# Suffixes, not substrings: "mlp.gate" also matches
# "switch_mlp.gate_proj" and would quantise the experts at 8 bits.
EIGHT_BIT = ("mlp.gate", "shared_expert_gate")


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from defaults import MTP
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=MTP, required=MTP is None,
                    help="folder with the downloaded mtp.safetensors")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--swap-gate-up", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out) if a.out else src / (
        f"mtp-{a.bits}bit{'-swapped' if a.swap_gate_up else ''}.safetensors")
    w = mx.load(str(src / "mtp.safetensors"))

    # 1. norm shift
    shifted = []
    for k in list(w):
        # "norm" anywhere, not a suffix: pre_fc_norm_embedding and
        # pre_fc_norm_hidden end in embedding/hidden, and missing them
        # is a wrong-weights bug that does not crash.
        if w[k].ndim == 1 and "norm" in k:
            w[k] = w[k] + 1.0
            shifted.append(k.replace("mtp.", ""))
    print(f"norm weights shifted by +1.0: {len(shifted)}")

    # 2. the experts into what SwitchGLU expects: [E, out, in] per projection.
    # Qwen3.6 ships them fused (gate_up_proj [E, 2*inter, hidden]) and has to
    # be split; Qwen3-Next-80B ships one tensor per expert and projection
    # (experts.N.gate_proj.weight) and has to be stacked, the same stack
    # mlx_lm's qwen3_next.sanitize does for the model's own layers, so no
    # gate/up convention is in question there.
    pre = "mtp.layers.0.mlp.experts."
    if pre + "gate_up_proj" in w:
        gu = w.pop(pre + "gate_up_proj")
        dn = w.pop(pre + "down_proj")
        mid = gu.shape[1] // 2
        first, second = gu[:, :mid, :], gu[:, mid:, :]
        gate, up = (second, first) if a.swap_gate_up else (first, second)
        print(f"experts split: gate/up {gate.shape}, down {dn.shape}"
              f"{'  (gate/up swapped)' if a.swap_gate_up else ''}")
    else:
        assert not a.swap_gate_up, "per-expert checkpoint: gate and up are named"
        n = 0
        while f"{pre}{n}.gate_proj.weight" in w:
            n += 1
        assert n, "no expert tensors under " + pre
        stacks = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacks[proj] = mx.stack([w.pop(f"{pre}{e}.{proj}.weight")
                                     for e in range(n)])
        gate, up, dn = stacks["gate_proj"], stacks["up_proj"], stacks["down_proj"]
        print(f"experts stacked: {n} x gate/up {gate.shape[1:]}, down {dn.shape[1:]}")
    w["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] = gate
    w["mtp.layers.0.mlp.switch_mlp.up_proj.weight"] = up
    w["mtp.layers.0.mlp.switch_mlp.down_proj.weight"] = dn

    # 3. quantise everything with a weight matrix, leave norms alone
    q, meta = {}, {}
    for k in sorted(w):
        v = w[k]
        if v.ndim < 2 or not k.endswith(".weight"):
            q[k] = v
            continue
        base = k[:-len(".weight")]
        bits = 8 if base.endswith(EIGHT_BIT) else a.bits
        wq, s, b = mx.quantize(v, group_size=GROUP, bits=bits)
        q[k] = wq
        q[base + ".scales"] = s
        q[base + ".biases"] = b
        meta[base] = {"group_size": GROUP, "bits": bits}
    mx.eval(q)

    nb = sum(v.nbytes for v in q.values())
    mx.save_safetensors(str(out), q)
    (out.parent / (out.stem + "-quant.json")).write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {out}")
    print(f"  {len(q)} arrays, {nb/1e6:.1f} MB "
          f"(from {sum(v.nbytes for v in w.values())/1e6:.1f} MB bf16)")
    for base in sorted(meta):
        print(f"  {base.replace('mtp.layers.0.', ''):44s} "
              f"{meta[base]['bits']} bits")


if __name__ == "__main__":
    main()
