"""llama.cpp's GGUF block formats on the CPU: types, tables and a numpy reference dequant (no MLX, so tools
and checks run on any machine without touching a GPU)."""
# The dequant is ggml's dequantize_row_* (ggml-quants.c, llama.cpp b1c902f0, MIT; THIRD_PARTY_NOTICES.md).
import numpy as np

# ggml type name -> (type id, elements a block, bytes a block): gguf-py constants.py GGML_QUANT_SIZES.
TYPES = {"Q4_0": (2, 32, 18), "Q4_1": (3, 32, 20), "Q5_0": (6, 32, 22), "Q5_1": (7, 32, 24), "Q8_0": (8, 32, 34),
         "Q4_K": (12, 256, 144),
         "Q5_K": (13, 256, 176), "IQ3_XXS": (18, 256, 98), "IQ4_NL": (20, 32, 18), "IQ2_S": (22, 256, 82),
         "IQ4_XS": (23, 256, 136), "Q2_0": (42, 64, 18)}
NAME_OF = {v[0]: k for k, v in TYPES.items()}

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int8)
# ggml's iq3xxs_grid (256 x 4 bytes) and iq2s_grid (1024 x 8 bytes), each byte an index into its value set
# packed 3 (iq3xxs) or 2 (iq2s) bits a byte, lowest byte first, matching
# gguf-py's own grids.
_IQ3XXS_VALS = (4, 12, 20, 28, 36, 44, 52, 62)
_IQ2S_VALS = (8, 25, 43)
_IQ3XXS_HEX = (
    "00000200400900b00f01001201902203b03d04104304804a05105505805a06106c07808008208408909009209909b09f"
    "0a90af0bd0c10c70c80ca0d50f810b11f12412f13b13d14114715a16a19d1b41c81cc1ce1e31f120120320820a211213"
    "21821a21c22722824024224925025228128328828a2912982ba2c02c22d02d92e62f63013053283503543663793853d2"
    "3e040040240940b41041241641942244144344544844a45145847347747848048248948f49049249f4a04ad4c14c84cc"
    "4f84fc51d52b54355756157c5c15c35ce5e560160860a61161362863563a6406426506596646666816836886956aa6ba"
    "6c96db71872773a74074675276d78c79e7b37db7f080480f81d81f82b82f87c89089f8a08b08b68c78e5904929934955"
    "9639789c59c89ca9d8a0aa21a38a40a46a56a6da8ca9aabaac2aebb08b13b17b3ab42b59ba8bd4be2c14c24c26c34c51"
    "c71c8fcb4cd8cded24d45d6ad9bdc3dd1e03e05e07e08e1ae2ae56e60e8aea5eaaec0ecdedbef0f11f21f40f42f54f98")
_IQ2S_HEX = (
    "0000000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100640066"
    "006900800082008500880091009400a000a500aa01010104010601090110011201150118012101240140014201450148"
    "01510154015601590160016501680181018401900192019501a101a40200020202050208021102140220022a02410244"
    "024602490250025502800285028a029402a2040104040406040904100412041504180421042404260429044004420445"
    "0448044a0451045404560459046004620465048104840486048904900495049804a104a40500050205050508050a0511"
    "051405160519052005250528054105440546054905500552055505580561056405800582058505880591059405a00601"
    "060406060609061006150640064506480651065406600681068406900800080208050808081108140816081908200825"
    "082a084108440846084908500852085508580861086408800885089408aa090109040910091209150918092109400945"
    "0948095109540960098109900a000a110a140a220a280a2a0a500a991001100410061009101010121015101810211024"
    "102610401042104510481051105410561059106010621065106810811084108610901095109810a110a4110011021105"
    "1108110a1111111411161119112011221125112811411144114611491150115211551158116111641180118211851188"
    "119111941201120412091210121512211224124012451251125412811284129014001402140514081411141414161419"
    "142014251428144114441446144914501452145514581461146414801482148514881491149414a01501150415061509"
    "151015121515151815211524154015421545154815511554156015811584159016001605160816111614162016411644"
    "1650168016aa180118041806180918101815181818211840184218451848185118541860188118841900190219051908"
    "191119141920194119441950196919a21a041a101a401a562000200220052008201120142016201920202025202a2041"
    "204420502052205520642080208a209420aa210121042110211221152121214021422145215121542160218121842190"
    "2200220a22222228222a224422502288228a22a824012404240624092410241524182421242424402442244524482451"
    "245424602481248424902500250525082511251425202541254425502566258026012604261026402659280028052811"
    "2814284128442850288a28aa29012904291029952a0a2a222a642a882a8a40014004400640094010401240154018401a"
    "4021402440264040404240454048404a40514054405640594060406240654081408440904095409840a140a441004102"
    "410541084111411441164119412041224125414141444146414941504152415541584161416441804182418541884191"
    "419441a04201420442104212421542184224424042454248425142544260428142844400440244054408440a44114414"
    "441644194420442244254428444144444446444944504452445544584461446444804482448544884491449444a04501"
    "4504450645094510451245154518452145244540454245454548455145544560456a4581458445904600460246054608"
    "461146144620464146444650468046a54801480448094810481248154818482148244840484248454848485148544860"
    "488448904900490249054908491149144920494149444950498049964a014a044a104a40500050025005500850115014"
    "501650195020502250255028504150445046504950505052505550585061506450805082508550885091509451015104"
    "510651095110511251155118512151245140514251455148515151545160518151845190520052055208521152145220"
    "524152445250526952805401540454065409541054125415541854215424544054425445544854515454546054815484"
    "549055005502550555085511551455205541554455505580560156045610562656405800580258055808581158145820"
    "584158445850585a588059015904591059405a005a195a855aa860016004600660106012601560186021602460406045"
    "604860516054606060846090610061026105610861116114612061416144615061806199620462106240625662a16400"
    "6405640864116414642064416444645064806501650465106540654a6568659266006694680168046810686568986900"
    "692a6a426aa1800080028005800880118014801980208025804180448050805280558058806180808085809180948101"
    "8104810981108112811581188121812481408142814581488151815481818184819081a982008205820a821182148241"
    "824482508401840484068409841084128415841884218440844284458448845184548460848184848490850085028505"
    "85088511851485208541854485508580858a86018604861086298640880088058811881488418844885088a289018904"
    "894089658a228a588a5a8a828aa290019004900990109012901590189024904090429045904890519054906090819084"
    "90909100910591119114914191449150915a920192049210924092a69400940294059408941194149420944194449450"
    "948094969501950495109540959895a19600964696649801980498109826984098a99900994999529a90a000a005a00a"
    "a014a022a02aa041a044a050a0a2a0aaa140a165a202a20aa222a228a22aa282a288a28aa2a8a401a404a410a440a489"
    "a4a4a500a519a651a80aa828a8a2a954a986aa08aa0aaa20aa22aa28aa88aaaa")


def _grid(hexs, vals, per, bits):
    """A packed grid as a uint8 array [entries, per]."""
    w = per * bits // 4
    codes = np.array([int(hexs[i:i + w], 16) for i in range(0, len(hexs), w)], np.uint32)
    idx = (codes[:, None] >> (np.arange(per, dtype=np.uint32) * bits)) & ((1 << bits) - 1)
    return np.array(vals, np.uint8)[idx]


IQ3XXS_GRID = _grid(_IQ3XXS_HEX, _IQ3XXS_VALS, 4, 3)
IQ2S_GRID = _grid(_IQ2S_HEX, _IQ2S_VALS, 8, 2)
# ggml's ksigns_iq2xs: seven sign bits and their parity as the eighth.
KSIGNS_IQ2XS = np.array([i | (bin(i).count("1") & 1) << 7 for i in range(128)], np.uint8)


def _f16(b, at):
    """The half at byte `at` of every block, as float32 (n,)."""
    return np.ascontiguousarray(b[:, at:at + 2]).view(np.float16)[:, 0].astype(np.float32)


def _u32(b, at, n=1):
    return np.ascontiguousarray(b[:, at:at + 4 * n]).view("<u4")


def _scale_min_k4(s):
    """ggml's get_scale_min_k4 for j = 0..7 of every block's 12 scale bytes: (sc, m) as float32 (n, 8)."""
    s = s.astype(np.uint8)
    sc = np.concatenate([s[:, 0:4] & 63, (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)], axis=1)
    m = np.concatenate([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], axis=1)
    return sc.astype(np.float32), m.astype(np.float32)


def _k_quant(b, five):
    """Q4_K (five False) and Q5_K: y = d * sc * q - dmin * m per 32-weight sub-block, as ggml rounds it."""
    n = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, m = _scale_min_k4(b[:, 4:16])
    d1 = d[:, None] * sc                          # (n, 8): sub-block j's scale and min
    m1 = dmin[:, None] * m
    ql = b[:, 48:176] if five else b[:, 16:144]
    ql = ql.reshape(n, 4, 32)
    q = np.stack([ql & 0xF, ql >> 4], axis=2).astype(np.int32)              # (n, 4, 2, 32): sub-block 2g + h
    if five:
        qh = b[:, 16:48].astype(np.int32)
        bit = (qh[:, None, None, :] >> (2 * np.arange(4)[None, :, None, None] + np.arange(2)[None, None, :, None])) & 1
        q = q + 16 * bit
    y = d1.reshape(n, 4, 2, 1) * q.astype(np.float32) - m1.reshape(n, 4, 2, 1)
    return y.reshape(n, 256)


def _q5(b, with_min):
    """Q5_0 (with_min False: (q - 16) * d) and Q5_1 (q * d + m) over 32-weight blocks."""
    d = _f16(b, 0)
    at = 4 if with_min else 2
    qh = _u32(b, at)[:, 0]
    qs = b[:, at + 4:at + 20].astype(np.uint32)
    j = np.arange(16, dtype=np.uint32)
    lo = (qs & 0xF) | (((qh[:, None] >> j) << 4) & 0x10)
    hi = (qs >> 4) | ((qh[:, None] >> (j + 12)) & 0x10)
    q = np.concatenate([lo, hi], axis=1).astype(np.int32)
    if not with_min:
        return (q - 16).astype(np.float32) * d[:, None]
    return q.astype(np.float32) * d[:, None] + _f16(b, 2)[:, None]


def _iq4(b, xs):
    """IQ4_NL (32-weight blocks, one d) and IQ4_XS (256, a 6-bit scale a 32 minus 32)."""
    n = b.shape[0]
    d = _f16(b, 0)
    qs = b[:, 8:136] if xs else b[:, 2:18]
    qs = qs.reshape(n, -1, 16)
    v = KVALUES_IQ4NL.astype(np.float32)
    q = np.concatenate([v[qs & 0xF], v[qs >> 4]], axis=2)                  # (n, sub, 32)
    if xs:
        sh = np.ascontiguousarray(b[:, 2:4]).view("<u2")[:, 0].astype(np.int32)
        sl = b[:, 4:8].astype(np.int32)
        ib = np.arange(8)
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
        dl = d[:, None] * (ls - 32).astype(np.float32)
    else:
        dl = d[:, None]
    return (dl[:, :, None] * q).reshape(n, -1)


def _sign(y, s):
    """y negated where bit j of s (n, ..., 1) is set, j the last axis of y (8 wide)."""
    neg = (s.astype(np.int32) >> np.arange(8)) & 1
    return np.where(neg == 1, np.float32(-1.0), np.float32(1.0)) * y


def _iq3_xxs(b):
    n = b.shape[0]
    d = _f16(b, 0)
    qs = b[:, 2:66].reshape(n, 8, 8).astype(np.int32)              # 8 sub-blocks, 8 grid indices each
    aux = _u32(b, 66, 8)                                          # (n, 8)
    db = d[:, None] * (np.float32(0.5) + (aux >> 28).astype(np.float32)) * np.float32(0.5)
    signs = KSIGNS_IQ2XS[(aux[:, :, None] >> (7 * np.arange(4, dtype=np.uint32))) & 127]   # (n, 8, 4)
    g = IQ3XXS_GRID[qs].reshape(n, 8, 4, 8).astype(np.float32)     # grid1 then grid2 of each l
    return _sign(db[:, :, None, None] * g, signs[..., None]).reshape(n, 256)


def _iq2_s(b):
    n = b.shape[0]
    d = _f16(b, 0)
    qs = b[:, 2:34].reshape(n, 8, 4).astype(np.int32)
    signs = b[:, 34:66].reshape(n, 8, 4)
    qh = b[:, 66:74].astype(np.int32)
    sc = b[:, 74:82].astype(np.int32)
    db = np.stack([sc & 0xF, sc >> 4], axis=2).astype(np.float32)               # (n, 8, 2)
    db = d[:, None, None] * (np.float32(0.5) + db) * np.float32(0.25)
    l = np.arange(4)
    idx = qs | ((qh[:, :, None] << (8 - 2 * l)) & 0x300)
    g = IQ2S_GRID[idx].astype(np.float32)                                       # (n, 8, 4, 8)
    return _sign(db[:, :, l // 2, None] * g, signs[..., None]).reshape(n, 256)


def _q4(b, with_min):
    """Q4_0 (with_min False: (q - 8) * d) and Q4_1 (q * d + m) over 32-weight blocks."""
    qs = b[:, 4:20] if with_min else b[:, 2:18]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=1).astype(np.int32)
    if not with_min:
        return (q - 8).astype(np.float32) * _f16(b, 0)[:, None]
    return q.astype(np.float32) * _f16(b, 0)[:, None] + _f16(b, 2)[:, None]


def _q8_0(b):
    return np.ascontiguousarray(b[:, 2:]).view(np.int8).astype(np.float32) * _f16(b, 0)[:, None]


def _q2_0(b):
    q = (b[:, 2:18, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    return (q.reshape(b.shape[0], 64).astype(np.int32) - 1).astype(np.float32) * _f16(b, 0)[:, None]


_REF = {"Q4_0": lambda b: _q4(b, False), "Q4_1": lambda b: _q4(b, True),
        "Q4_K": lambda b: _k_quant(b, False), "Q5_K": lambda b: _k_quant(b, True),
        "Q5_0": lambda b: _q5(b, False), "Q5_1": lambda b: _q5(b, True), "Q8_0": _q8_0, "Q2_0": _q2_0,
        "IQ4_NL": lambda b: _iq4(b, False), "IQ4_XS": lambda b: _iq4(b, True), "IQ3_XXS": _iq3_xxs,
        "IQ2_S": _iq2_s}


def ref_dequant(raw, t):
    """float32 weights of `raw` (bytes or uint8 array) in ggml type t (a name of TYPES), ggml's
    dequantize_row_* without fused multiply-adds."""
    _, be, bb = TYPES[t]
    b = np.frombuffer(raw, np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw, np.uint8)
    assert b.size % bb == 0, (t, b.size)
    with np.errstate(over="ignore", invalid="ignore"):
        return _REF[t](b.reshape(-1, bb)).reshape(-1).astype(np.float32, copy=False)


# Byte offsets of each type's f16 fields in a block (d, and dmin or m).
HALVES = {"Q4_0": (0,), "Q4_1": (0, 2), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q5_1": (0, 2), "Q5_0": (0,), "Q8_0": (0,),
          "Q2_0": (0,),
          "IQ4_NL": (0,), "IQ4_XS": (0,), "IQ3_XXS": (0,), "IQ2_S": (0,)}


def random_experts(rng, n, rows, length, t, scale=0.01):
    """n random experts [n, rows, row bytes] of type t with finite scales (~scale), as numpy uint8."""
    _, be, bb = TYPES[t]
    b = rng.integers(0, 256, (n * rows * length // be, bb), dtype=np.uint8)
    for at in HALVES[t]:
        b[:, at:at + 2] = (rng.standard_normal((len(b), 1)) * scale).astype(np.float16).view(np.uint8)
    return b.reshape(n, rows, length // be * bb)
