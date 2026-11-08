# How it works

This page explains what the engine does with a model and why it is built this way, in plain words. It assumes no background in machine learning. Every number here is in `research/` with its machine, setup and date.

## 1. The problem: the model does not fit

A language model is a large collection of numbers called weights. When you send a prompt, the program multiplies your text by those weights, layer by layer, and out comes one token: a word or a piece of a word. Then it does the same thing again for the next token.

The useful models are much bigger than the memory of an ordinary computer. The tested Qwen3.6-35B-A3B has 35 billion parameters; the tested Qwen3.8-Flash-Next is larger still. Keeping all of a model's weights in RAM is what most programs do, and it is why they ask for a data-centre card.

This engine takes a different route: most of the weights stay on the SSD, and only the small part a token needs is read into memory.

## 2. Mixture of experts

Qwen3.6-35B-A3B is a mixture-of-experts model. Instead of one large network in each layer, a layer holds many small networks called experts, plus a router. The router looks at the token and picks a few experts for it. The 35B has 40 layers and 256 experts in each; the router picks 8. Flash-Next has 48 layers and 512 experts in each; its router picks 10.

Only the chosen experts do any work for that token. The rest are idle. This is the opening for a small computer: if only 8 of 256 experts are needed, the program does not need all 256 in memory at once.

The part every token needs is small enough to stay resident: attention weights, the routers, the token embeddings and the shared expert. In the tested setups that is the part that never leaves RAM. The expert weights are the part measured in tens of gigabytes.

## 3. One token, step by step

1. The token arrives at layer 1.
2. The router scores all experts and chooses the top few.
3. For each chosen expert, the engine looks in its in-memory expert cache.
4. A hit is used as it is. A miss is read from the SSD: the expert's weights are stored as one record, so the read is a few large reads, not thousands of tiny ones.
5. The GPU multiplies the token by each expert's weights and adds the results.
6. The output goes to the next layer, and the process repeats. After the last layer, the model produces one token, and the whole path starts again for the next token.

```text
prompt tokens
     |
     v
+---------------------------------------------+
| layer 1                                     |
|   router: 8 of 256 experts (35B)            |
|   +-------------------------------------+   |
|   | expert cache (RAM)                  |   |
|   |   hit: use it                       |   |
|   |   miss: read one record from SSD    |   |
|   +-------------------------------------+   |
|   GPU: multiply, add                        |
+---------------------------------------------+
     |
     v
+---------------------------------------------+
| layer 2 ... layer 40 or 48                  |
+---------------------------------------------+
     |
     v
one new token
```

At the same time, the engine reads ahead for the next layer: it guesses which experts that layer will ask for and starts those reads while the GPU is still working on the current layer. A wrong guess costs a read that is not used; a right guess hides the read behind math.

## 4. The expert cache

The cache is a set of expert slots in memory. Its size is chosen from the memory actually free: too small and too many tokens wait for the SSD; too large and the computer starts to swap.

Two effects make a modest cache useful:

- **Popular experts.** Routing is uneven. Some experts are chosen far more often than others, so a cache that holds the popular ones serves a large share of requests.
- **Reuse across tokens.** Neighbouring tokens and nearby text often choose similar experts.

On the 16 GB M4 with Flash-Next at a 6 GB cache, 53% to 66% of the experts a token asked for were served from memory, and 64% of the token's time was spent waiting for the rest to arrive from the drive (565 MB per token at about 3.4 GB/s, 2026-10-11). Those numbers are why the cache size matters so much.

The size is not a fixed number in the code. The launcher measures the machine, and the engine grows the cache while memory is free and gives memory back when the system comes under pressure.

## 5. Reading the prompt

Before the first reply token can appear, the whole prompt has to go through the model. This is called prefill. It is a pass over all prompt tokens, layer by layer, and it reads far more experts than a single decode token does.

The engine plans the pass from the memory it has. If a prompt fits in one pass, it runs as one pass; if not, it is cut into the fewest equal pieces that fit, and conversation state is kept across the pieces. The first token is therefore the slowest part of a long request:

| prompt | computer | first token | date |
|---|---|---|---|
| 16k tokens, 35B (mixed research pack) | 16 GB M4 MacBook Pro | 45.6 s | 2026-10-07 |
| 16k tokens, Flash-Next | 16 GB M4 MacBook Pro | 136.2 s fresh, 137.5 s warm | 2026-10-25 |
| 16k tokens, Flash-Next | Linux PC, RTX 5060 Ti 16 GB | 34.4 s fresh, 31.4 s warm | 2026-10-26 |
| 19 to 54-token question, Flash-Next | 16 GB M4 MacBook Pro | 2.5 to 4.5 s | 2026-10-14 |

After the first token, each further token is much cheaper because only the new token goes through the layers, and because the conversation's state stays in memory.

## 6. Where the time goes

A decode token is a chain of small GPU operations and, between them, waits for expert records. On the 16 GB M4 with Flash-Next's EXL3 pack on one drive, a Russian token took 90.7 ms: 39.2 ms of reads the GPU waited for and 51.6 ms of the GPU's own work on the experts, attention and the small glue operations (2026-10-21). With the experts on two drives the same prompt decoded at 13.8 tokens/s, about 72 ms a token (2026-10-21). On the RTX 5060 Ti PC at its 16 GB budget, a Russian token took 32.2 ms on the GPU clock: 20.3 ms of GPU work and 8.9 ms parked on reads (2026-10-26).

This is why the SSD is the main lever, and why the engine spends its effort on three things: reading fewer bytes, reading them earlier, and using the bytes it already has.

Measured read speeds on the test machines:

| drive | speed | date |
|---|---|---|
| internal SSD, 16 GB M4 MacBook Pro | 3.42 GB/s | 2026-10-26 |
| external Thunderbolt SSD, same computer | 3.38 GB/s | 2026-10-13 |
| two drives together, same computer | 7.02 GB/s on a 1 GB region; about 6.5 GB/s over whole files (2026-10-19) | 2026-10-13 |
| one NVMe SSD, RTX 5060 Ti Linux PC | 4.35 GB/s in random 1 MB reads | 2026-10-17 |

## 7. Two drives and smaller experts

Flash-Next's published 4-bit pack has 75.5 GB of expert weights. The recommended pack stores the same experts in a 3-bit trellis format (EXL3) that is 45.8 GB, a lossy change described in the quality section of the README.

With the experts split across the internal SSD and an external Thunderbolt SSD, Flash-Next decoded at 13.8, 14.2 and 13.0 tokens/s in Russian, English and Python on the 16 GB M4 (2026-10-21). Using both drives was worth 1.107x over the internal drive alone, and the 3-bit pack was 1.58x the 4-bit pack on one drive and 1.19x the two-drive 4-bit layout.

The engine reads the two files as one stream of expert records, so the model does not have to fit on a single disk.

## 8. Dense models are different

A dense model, such as Qwen3-4B or Qwen3-8B, has no experts to skip: every weight is needed for every token. These models run entirely in memory; the SSD is read when the model loads, not while it decodes.

They still get faster in two ways:

- **A small draft model.** A same-family 0.6B model guesses several tokens; the large model checks them in one pass and keeps the ones it would have chosen itself. In a 300-token benchmark on the 16 GB M4, Qwen3-4B went from 42.3 to 55.2-66.8 tokens/s (best 1.58x) and Qwen3-8B from 23.1 to 35.7-46.8 (prose 1.55x, code 2.03x), with the reply identical up to exact bf16 ties (2026-09-12).
- **Batching.** Several requests in flight decode as rows of one forward pass. Dense models can therefore serve a few users at once; a mixture-of-experts model answers one request at a time.

## 9. The server and conversations

`server.py` is a small HTTP server with an OpenAI-compatible interface. It serves the chat page and the API from the same process and the same model. One server serves one model.

A conversation's prompt is kept after a reply. When the next message arrives, the server reads only the new part and reuses the earlier state, which is why the second message of a chat is much faster than the first. It keeps one conversation by default; the `--keep-convs` flag keeps more for agents that switch between tasks.

Requests from other computers are refused by default. The server listens on the loopback address, so it is reachable from this computer's browser and editor, not from the network.

## 10. What the launcher measures

The launcher does not guess from a model name or a marketing number. On first use it measures:

- the operating system, the chip, the RAM and the GPU, including the GPU's read bandwidth on a decode-shaped 4-bit matmul;
- the read speed of the disk the models will live on, using expert-sized random reads past the page cache;
- the free memory and free disk, read again on every start.

Those measurements become the flags the server starts with: the size of the expert cache, the prefill budget, and whether the native reader can be used. The speed shown in the menu is an estimate from the same inputs, and it landed within 0.88x to 1.23x of the measured decode speed at every point it could be checked against on the 16 GB M4 and the RTX 5060 Ti PC (2026-10-26).

The profile is cached and measured again when the hardware, the operating system or the MLX version changes.

## 11. What is not here

- No training and no fine-tuning. The engine runs the weights a publisher shipped.
- No model-specific training. The same engine runs every model; the launcher chooses among its own measured strategies from the model's architecture and from measurements on this machine.
- No cloud service. Weights are downloaded once, and every reply is computed on your computer.
- No hidden quality switch. Every option and default that can change a reply, even only by rounding, is named in the README's quality section with the check behind it.
