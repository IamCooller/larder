# Claude Code on a local model

`server.py` also speaks Anthropic's Messages API (`POST /v1/messages`, `POST /v1/messages/count_tokens`),
so Claude Code, and any program built on an Anthropic SDK, can use the model you serve by setting
`ANTHROPIC_BASE_URL`. The OpenAI API and the chat page keep working beside it; start the server as
described in [VSCODE.md](VSCODE.md) (the address below is the default port, 8765).

## Run it

```
python3 server.py --model /path/to/model --keep-convs 4
```

then, in another terminal, in the project you want to work on:

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=local
export ANTHROPIC_DEFAULT_HAIKU_MODEL=local
export ANTHROPIC_SMALL_FAST_MODEL=local
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768
claude --bare
```

| Variable | Why |
|---|---|
| `ANTHROPIC_BASE_URL` | the server; write `127.0.0.1`, not `localhost` (the server listens on IPv4 only) |
| `ANTHROPIC_AUTH_TOKEN` | any text; Claude Code wants a credential. The server checks it only when it was started with `--api-key`: then put that key here (Claude Code sends it as `Authorization: Bearer`) |
| `ANTHROPIC_MODEL` | any name: there is one model per server and the name is ignored (`/v1/models` shows the real one) |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` | Claude Code sends its small side requests (titles and the like) to a "fast" model; this keeps them on the same server. The second name is deprecated in favour of the first, and setting both is harmless |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | no update checks, telemetry or other calls to Anthropic's hosts |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | leaves out the client-version block that Claude Code puts at the start of the system prompt; before Claude Code 2.1.181 that block changed on every request, which made the server read the whole prompt again each turn, and later versions keep it stable but still change it per conversation |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | the window to work in. Claude Code does not know the model name, warns that it assumes 200k tokens and compacts at that size; give it the context you are willing to read (a longer prompt is slower) |

The same variables can go in the `env` block of `~/.claude/settings.json` (or a project's
`.claude/settings.local.json`) instead of the shell. `/status` inside Claude Code shows the base URL it uses.
Sources: [environment variables](https://code.claude.com/docs/en/env-vars),
[LLM gateways](https://code.claude.com/docs/en/llm-gateway-connect),
[gateway compatibility](https://code.claude.com/docs/en/llm-gateway-protocol).

### Why `claude --bare`

`--bare` skips hooks, skills, plugins, MCP servers, auto memory and `CLAUDE.md`, and leaves the Bash, file
read and file edit tools. Its first request is small; the default one is not, and a local model reads every
token of it before the first answer. Measured on 2026-10-27 on a Mac, against the server's test stub (no model;
the size does not depend on the machine), with Claude Code 2.1.310 and an empty settings folder, the JSON
Claude Code sent for the first message "say hi":

| | system prompt | tools | messages | total |
|---|---|---|---|---|
| `claude --bare` | 0.3 KB | 3 (Bash, Edit, Read), 3.5 KB | 0.1 KB | about 4 KB, roughly 1 thousand tokens |
| `claude` | 6 KB | 12, 30 KB | 10 KB | about 46 KB, roughly 12 thousand tokens |

Your own `CLAUDE.md`, skills and MCP servers come on top of the second row, which is how a first turn reaches
tens of thousands of tokens. Use plain `claude` when you need those, and expect the first turn to be slow.

### Flash-Next as a background coding agent (16 GB GPU, CUDA)

What to expect, measured on ten bug and feature tasks of this project's own repository on an RTX 5060 Ti
16 GB, in two setups:

- Claude Code (`claude --bare -p`, a one-hour box, one run a task), the 3-bit EXL3 experts at cache bias 0.5,
  2026-10-28: 4 of the 8 counted tasks passed, with the first edit 7 to 52 minutes in. A hosted model in the
  same setup passed 10 of 10, each in under a minute (2026-10-29).
- This project's own agent harness (25 replies and 20 minutes a run, two runs a task, cache bias 0),
  2026-11-03 to 2026-11-07: the 4-bit EXL3 experts (K4) passed 9 of 20 runs, with an older tool-call parser;
  K8T, the same experts on an 8-bit trunk, passed 15 of 20; a hosted model through the same harness passed
  17 of 20.

K4 and K8T are research packs that the launcher does not install. The launcher installs the 3-bit EXL3 pack
and offers the published 4-bit pack as a variant. Give it well-scoped tasks to run unattended and read every
diff. The 3-bit experts did worse (on one task they failed in 4 of 5 setups where both 4-bit packs passed,
2026-10-30), and cache bias 0.5 misquotes edits, so run it at exact routing with the model's own sampling:
take the server command `./larder --dry-run` prints for Flash-Next and add the two flags.

```
python3 server.py --model /path/to/flash-next <the launcher's flags> --cache-bias 0 --model-sampling
```

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=local
export ANTHROPIC_DEFAULT_HAIKU_MODEL=local CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 CLAUDE_CODE_MAX_OUTPUT_TOKENS=16000
export API_TIMEOUT_MS=1800000 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
claude --safe-mode -p "$(cat task.md)" --tools Bash,Edit,Read,Write --append-system-prompt-file CLAUDE.md
```

- `--model-sampling`: Claude Code sends no temperature, and the greedy runs looped (one task took 67 steps
  greedy under `--bare`, 12 with the model's sampling and the full prompt).
- `--safe-mode` keeps Claude Code's full system prompt without this machine's hooks and plugins; the Write tool
  saves the model from creating files through Bash.
- `MOE_KV_HOST=1` in front of the server command (CUDA only) keeps long contexts fast (decode at 64k 114 -> 91 ms
  a token at `--cache-bias 0`); it changes no byte of the output.
- `--cache-bias 0` is the exact routing: no edit failed on misquoted old text there (one to five a task at 0.5),
  but decode is about half as fast (~10 against ~20 tokens a second), and the pass rate was no better.
- Thinking (`--adaptive-thinking`) made each step about twice as long; with `--thinking-budget 2048` it passed a
  task it had lost to 38 minutes of thinking, and still missed two others. The budget changes the model's output,
  so it is opt-in.

## What works

Claude Code runs its tools itself; the model only has to ask for them. The server puts Claude Code's tool list
into the model's chat template, reads the call back out of the reply (`tools.py`) and returns it as a
`tool_use` block; Claude Code runs the tool and sends the result back as a `tool_result`, which becomes a
`tool` turn in the prompt.

- Streaming, with a `ping` every 5 seconds while the prompt is read, so Claude Code's idle timers see traffic.
- Tool use, `tool_choice` `auto` and `none` (`any` and `tool` are read as `auto`: a call cannot be forced).
- Thinking: Claude Code always asks for `{"type": "adaptive"}` (`MAX_THINKING_TOKENS` does not change that), which
  the server answers with thinking off unless it was started with `--adaptive-thinking`; `{"type": "enabled"}`
  turns it on either way (then `thinking` blocks come back, and count against `max_tokens`). The `thinking`
  blocks Claude Code sends back with earlier turns go into the prompt as those turns' `reasoning_content`, so
  the template renders each step as it was generated, and `output_config.effort` becomes the template's
  `reasoning_effort`. `--thinking-budget N` caps a thinking turn at N tokens: the server then writes Qwen's
  early-stop text ("Considering the limited time by the user, I have to give the solution based on the thinking
  directly now." and `</think>`, from the
  [Qwen docs](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html)) into the thinking block
  and the model answers; a request's `budget_tokens` does the same, the smaller of the two winning. A turn
  with a budget decodes the plain way, without an MTP head, a draft or forced rows, and alone on a dense
  model's batch.
- Images in messages and in tool results, when `/healthz` says `"vision": true`.
- `stop_sequences`, `max_tokens`, and the cache: the next request of a conversation sends the same beginning
  again and the server reads only what was added. `usage.cache_read_input_tokens` is what came from the cache;
  `usage.input_tokens` is what was read. `--keep-convs 4` keeps the main conversation beside the side
  requests, which have their own prompt.
- `/v1/messages/count_tokens` renders system, messages and tools through the model's own chat template and
  counts them with its tokenizer, so `/context` shows real numbers.
- `anthropic-version` and `anthropic-beta` are accepted and ignored. `x-api-key` and `Authorization` are
  checked only when the server was started with `--api-key`, as on the OpenAI route: then one of them must
  carry the key. Without `--api-key` they are ignored, and the Origin and Host guard of the OpenAI route
  applies unchanged.
- A request whose client hangs up stops at its next token, streamed or not, and one that gave up while queued
  never starts, so Claude Code's retries of a slow request no longer queue behind the abandoned ones (a
  compaction summary at 16 tok/s was once resent six times).

| Request field | What the server does |
|---|---|
| `model`, `metadata`, `service_tier`, `context_management`, `cache_control` | accepted, ignored |
| `thinking` | `enabled` turns thinking on, its `budget_tokens` capping it (the smaller of it and `--thinking-budget`); `adaptive` is on under `--adaptive-thinking`, else off; `disabled` is off |
| `output_config` | with thinking on, `effort` becomes the template's `reasoning_effort`: `low` and `medium` as they are, `high`, `xhigh` and `max` as `xhigh`; the other fields are ignored |
| `temperature`, `top_p`, `top_k` | used only when the server was started with `--sampling`, as on the OpenAI route |
| `system` | a string or text blocks, joined with newlines |
| `messages` | `user` and `assistant`; a `system` entry before the first reply (Claude Code's environment block) joins the system prompt, since chat templates take one leading system turn, and a later one becomes a user turn inside `<system-reminder>` |
| content blocks | `text`, `image`, `tool_use`, `tool_result`, `thinking` (an assistant turn's `reasoning_content`) and `redacted_thinking` (dropped); others (`document`, ...) are a 400 |
| `tools` | entries with a `name` and an `input_schema`, except those marked `defer_loading` (Claude Code's placeholder); the rest are left out |

Errors use Anthropic's shape, `{"type": "error", "error": {"type", "message"}}`, with status 400
(`invalid_request_error`), 413 (`request_too_large`), 500 (`api_error`, generation failed) and 503 (no model
loaded). A failure after a stream has started arrives as an `error` event.

## Limits

- The first turn is slow. Claude Code sends its whole prompt and tool list, thousands of tokens even with
  `--bare` and tens of thousands with your own `CLAUDE.md` and MCP servers; the model has to read them before
  the first word. Later turns of the same conversation read only what was added. If a first turn needs more
  than 5 minutes, raise `CLAUDE_STREAM_IDLE_TIMEOUT_MS` (Claude Code allows up to 30 minutes) so it does not
  give up on a quiet stream.
- With tools in the request, which is every Claude Code request, the reply is collected whole and then sent as
  blocks, so the text does not appear token by token (the same as the OpenAI route with tools).
- A local model calls tools less reliably than Claude does: wrong parameters, a call written as text, or no
  call at all are more likely, and the more tools there are, the worse it gets. A larger model, `--bare`, or
  `--tools "Bash,Edit,Read"` to shrink the list helps.
- Claude Code asks for up to 32000 output tokens. That is a ceiling; the reply stops at the model's own end.
- Web search, web fetch, computer use and the other Anthropic-run tools do not exist here; the same for
  `document` (PDF) blocks.
- A request stopped by `stop_sequences` has no engine token count; its `output_tokens` is the number of text
  pieces read.

## Checked

On 2026-10-27, with Claude Code 2.1.310 against the test stub of `tests/test_webchat.py` (no model): a plain reply, and a turn
where the stub asked for a Bash call, Claude Code ran `echo hi` and sent the result back, ending the
conversation in two turns. `tests/test_anthropic_api.py` checks the API itself. With Flash-Next's 3-bit
EXL3 experts at cache bias 0.5 on an RTX 5060 Ti 16 GB, `claude --bare -p` worked ten real tasks of this
repository in a one-hour box each (2026-10-28): 4 of the 8 counted passed.
