# Using the server from VS Code, and the browser chat

`server.py` speaks the OpenAI chat API, so VS Code extensions that accept an
"OpenAI-compatible" endpoint work with it. The same server also serves a chat
page, with its CSS, JS and vendored libraries, at its own address. It speaks Anthropic's Messages API
too, so Claude Code can use it: see [CLAUDE_CODE.md](CLAUDE_CODE.md).

Every address below uses port 8765 (the default). If you start the server with
`--port 9000`, use 9000. Write `127.0.0.1`, not `localhost`: some clients resolve
`localhost` to the IPv6 address first, and the server listens on IPv4 only.

When the launcher (`./larder` or the full-screen menu) starts the server, it always uses one chat
port (8800 by default), the same for every model and every computer: the browser keeps its saved
chats, and a bookmark keeps working, when you switch models.

## Start the server

```
python3 server.py --model /path/to/model
```

When it prints `serving on http://127.0.0.1:8765`, the model is loaded.

- Chat in the browser: open http://127.0.0.1:8765/ (streamed replies, markdown
  with highlighted code, a Thinking switch with its seconds, image attach when the
  model can see, a bar for how full the model's context is with Compact and Clear,
  Regenerate, Edit and resend, and saved conversations in a sidebar; every asset
  is local and nothing is loaded from the network). Its header has a live pill
  (decode speed and memory) that opens the Status page at `/monitor`, and a link
  to Settings at `/settings`: the model in use and its speed, the launcher's
  models with their variants, the memory limit and the engine's options, where
  the chats are stored with Export and Import, and what the computer is, so no
  command-line flag is needed. Keys: Enter sends, Shift+Enter is a new line, Esc
  stops a reply, Ctrl or Cmd+K starts a new chat.
- Check the API: `curl http://127.0.0.1:8765/v1/models`

The model name an extension sends is ignored (there is one model per server), so
any name works in the settings below.

## Continue

[Continue](https://docs.continue.dev) reads `~/.continue/config.yaml`. Add a model:

```yaml
name: Local
version: 0.0.1
schema: v1
models:
  - name: Local model
    provider: openai
    model: local
    apiBase: http://127.0.0.1:8765/v1
    apiKey: local
    roles:
      - chat
      - edit
      - apply
    defaultCompletionOptions:
      contextLength: 16384
      maxTokens: 2048
    requestOptions:
      timeout: 600000
```

- `apiKey` can be any text; the server checks it only when started with `--api-key`, and then it must be that key.
- `timeout` is in milliseconds. A long prompt takes a while to prefill before
  the first token, so give it more than the 60 s used in Continue's examples.
- Agent mode (tools): add `capabilities: [tool_use]` to the model. The server
  passes the tool list to the model's chat template and turns the model's call
  back into an OpenAI `tool_calls` reply, so how well it works depends on the
  model. A reply to a request with tools arrives whole, not token by token.
- Tab autocomplete is not recommended: it sends many small fill-in-the-middle
  requests that need low latency, which this engine is not built for, and the
  server has no `/v1/completions` endpoint.

## Cline

In the Cline panel open Settings (gear icon) and set:

| Field | Value |
|---|---|
| API Provider | OpenAI Compatible |
| Base URL | `http://127.0.0.1:8765/v1` |
| API Key | any text, for example `local` |
| Model ID | `local` |
| Context Window Size | `16384` (raise it if your machine has the memory) |
| Max Output Tokens | `4096` |
| Image Support | off (on only if `/healthz` says `"vision": true`) |
| Computer Use | off |

Cline's own prompt is many thousand tokens long, so the first step of a task is
slow while it is read. The next steps of the same task send the same beginning
again, and the server only reads what was added (`--keep-convs N` keeps more than
one conversation if you work in two tasks at a time).

## GitHub Copilot Chat (custom model)

Copilot Chat can use your own endpoint without a Copilot plan; on Business or
Enterprise an administrator has to allow "Bring Your Own Language Model Key in
VS Code". Source: [VS Code, language models](https://code.visualstudio.com/docs/copilot/customization/language-models).

1. Command Palette, `Chat: Manage Language Models`.
2. `Add Models`, then `Custom Endpoint`. Enter a group name, a display name and
   any API key. Choose the API type `Chat Completions`.
3. In the `chatLanguageModels.json` that opens, use:

```json
[
  {
    "name": "Local",
    "vendor": "customendpoint",
    "apiKey": "local",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "local",
        "name": "Local model",
        "url": "http://127.0.0.1:8765/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 16384,
        "maxOutputTokens": 4096
      }
    ]
  }
]
```

- Give the full `.../v1/chat/completions` URL; VS Code uses it as it is.
- `vision`: set it to `true` only when `/healthz` says `"vision": true`; the server
  answers 400 to an image sent to a model that cannot see.
- `toolCalling: true` is needed for the model to appear for agents in chat.
  Tool calls are read from the model's text (see Continue above), and a reply
  to a request with tools arrives whole.
- `maxInputTokens` plus `maxOutputTokens` must fit the context you are willing
  to use; a longer prompt takes longer to read.
- Older VS Code Insiders builds call this provider "OpenAI Compatible"
  (`"vendor": "customoai"`, and the older setting `github.copilot.chat.customOAIModels`).
  The model fields are the same. Current builds deprecate both in favour of
  Custom Endpoint.

## What the server supports

| | |
|---|---|
| `GET /v1/models` | the loaded model (its id is the model folder's name, with `vision` true or false) |
| `GET /healthz` | `{ok, model, loaded, vision}` |
| `GET /status` | the server's counters since it started, see [Status](#status) |
| `POST /v1/chat/completions` | `stream` true or false; `messages` with string content or lists of text parts, and `image_url` parts for a model that can see ([Images](#images)); `max_tokens` or `max_completion_tokens` (default 512 if neither is sent); `chat_template_kwargs.enable_thinking` ([Thinking](#thinking)); `tools` and `tool_calls` as above; `response_format`; `stream_options.include_usage` |
| Reply | `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) in a non-stream reply, and in the last stream chunk when `include_usage` is set; `finish_reason` is `stop`, `length` (the token limit was reached) or `tool_calls`; `reasoning_content` next to `content` when thinking is on |
| Errors | `{"error": {"message", "type"}}` with status 400, 403, 404, 413 (a request over 64 MB) or 503 |
| Ignored | `stop`, `n`, `logprobs`, `tool_choice`; `temperature`, `top_p`, `top_k`, `min_p` and `seed` count only when the server was started with `--sampling` |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic's Messages API, with streaming, tools, images and thinking, for Claude Code and Anthropic SDK clients; errors in Anthropic's shape; see [CLAUDE_CODE.md](CLAUDE_CODE.md) |
| Not there | `/v1/completions`, embeddings, several models at once |

### Thinking

Both shipped chat templates read `enable_thinking`. A request turns thinking on with
`"chat_template_kwargs": {"enable_thinking": true}` (the key is also read at the top level of
the body; `chat_template_kwargs` wins when both are there). A request that says nothing, or
`false`, gets what it always got: the template's empty think block, so the model answers
straight away.

With thinking on, the reply's thinking comes back apart from the answer, the way other
OpenAI-compatible servers (vLLM, llama.cpp) send it:

- non-stream: `choices[0].message.reasoning_content` (text before `</think>`) and `content` (the answer);
- stream: `delta.reasoning_content` pieces first, then `delta.content` pieces; a request with
  `tools` arrives whole, as before, and carries both;
- a reply that hits `max_tokens` while still thinking has `reasoning_content`, an empty
  `content` and `finish_reason: "length"`.

The thinking counts against `max_tokens` (the chat page sends 8192 when its Thinking switch is
on, 2048 otherwise), and a thinking reply is longer, so it comes later. Turning thinking on or
off changes the start of the prompt in Flash-Next's template (it adds a reasoning-effort line to
the system message), so the first request after a switch reads the prompt again. Send
`reasoning_content` back on an assistant message only if you want the template to show it to
the model; clients that do not send it get the usual empty think block in the history.

### Images

A message's `content` may be a list that mixes text parts with
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` parts. The server
keeps the order and gives the list to the engine as it is, when the loaded engine can see
(`"vision": true` in `/healthz`, `/v1/models` and `/status`).

- Only `data:` URLs, base64, of PNG, JPEG, WebP or GIF (a GIF is read as its first frame).
  The type is checked against the file's first bytes. A remote `http(s)` URL is refused: the
  server never fetches one.
- At most 10 MB per image (decoded) and 64 MB per request: 400 and 413 otherwise, with the
  reason in `error.message`.
- A model without a vision part answers 400, `this model has no vision part; images are not
  supported`, to any request with an image part, and the chat page greys out its attach button.

### Status

`GET /status` returns what the server has done since it started:

| Field | |
|---|---|
| `model`, `loaded`, `vision` | the loaded model's path, whether one is loaded, whether it takes images |
| `uptime_s` | seconds since the server started |
| `requests` | replies finished (a request that failed, or whose client hung up, is not counted) |
| `prompt_tokens`, `cached_tokens` | prompt tokens of those replies, and how many of them came from a kept cache instead of being read again (`cached_tokens` is part of `prompt_tokens`; a dense model's replies that shared a forward pass with another count none) |
| `completion_tokens` | tokens written |
| `first_token_mean_s`, `first_token_last_s` | seconds from a request's arrival to its first token (waiting for the model included), the mean over all replies and the last one; `null` before the first reply |
| `decode_tok_s` | tokens after the first per second of decoding, over all replies together (so a long reply weighs more than a short one); `null` before a reply of two tokens |
| `memory_gb`, `memory_peak_gb` | the process's memory footprint now and at its highest (on a CUDA machine, the GPU memory in use) |
| `store_mb` | under `--mem-peak-gb`, the size of the expert store after the last reply; `null` otherwise |
| `context_length` | tokens the model reads at most, from its `config.json` (`max_position_embeddings`, at the top or under `text_config`); `null` when the config does not say |
| `memory_budget_gb`, `memory_budget_max_gb` | the engine's memory limit (`--mem-peak-gb`) and the ceiling it may grow to (`--mem-peak-max-gb`); `null` when none is set |
| `host`, `machine`, `started_at` | the computer's name, its `chip`, `ram_gb`, `os`, `backend`, `gpu`, `cpus` and `mlx`, and the Unix time the server started |

`GET /admin/history` (local only, like the other `/admin` routes) returns `{replies, capacity}`:
the last 200 replies, oldest first, each with `t` (Unix seconds), `ttft_s`, `decode_tok_s`,
`prompt_tokens`, `cached_tokens`, `completion_tokens` and `memory_gb`. The server keeps them in
memory only.

The Status page at `/monitor` draws these: tiles for the speed, the first token, the memory against
its limit and the uptime, a chart of the decode speed and one of the time to first token for each
reply, the tokens, the machine, and the replies as a table. The chat's header pill shows the
speed and the memory, and reads `/status` again after every reply. `GET /stats` is a different
thing: the engine's report on the last turn alone.

### Settings

The page's Settings view is these four endpoints, local-only like `/admin/load` (the same
`Origin` and `Host` guard, 403 otherwise):

| Endpoint | |
|---|---|
| `GET /admin/models[?refresh=1]` | the launcher's list for this computer: `models[]` with `id`, `name`, `kind`, `about`, `license`, `tested`, `unavailable`, `not_made[]`, and `variants[]` with `key`, `label`, `note`, `size_bytes`, `tps` (the estimate here), `ok` / `why`, `status`, `ready`, `make_bits`, `path`, `memory`, `current`; also `current` (the model in use, with `tps` from `/status`), `settings`, `machine`, `job` and `building`. The list is built off the request thread, so the first answer may say `building: true` |
| `POST /admin/prepare` `{model, variant}` | starts the launcher's own `launch.py --prepare` as a child in its own session, so a cancel stops the packer too; 409 while one runs, 404 for a variant the launcher does not list |
| `GET /admin/prepare` | that child: `state` (`idle` / `running` / `done` / `failed` / `cancelled`), `bytes` against `total`, `rate_bps`, `eta_s`, `seconds`, `log[]` and `error` |
| `DELETE /admin/prepare` | TERM to the child's session, then `state: cancelled`; the same choice resumes where it stopped |

Switching a model is the existing `POST /admin/load` with `{"model": path, "plan": true}`: the
launcher's own flags for that model, for the parts a swap can set (its memory budget, its cache
bias and that bias's budget). The answer carries `plan.set_now` and `plan.at_start`, the flags
the running server can only take when it starts (`--prefill-step`, `--keep-convs`), which the
page shows when they differ. The page's options are the same call with explicit keys:

| The page's words | The key | What the engine gets |
|---|---|---|
| Memory limit: Auto | `"memory": "auto"` | the launcher's plan for this model and this machine now (`--mem-peak-gb`, `--mem-peak-max-gb`); a model it does not list keeps the server's own |
| Memory limit: a number | `"memory": 9.5` | `--mem-peak-gb 9.5`, no ceiling |
| Faster replies | `"fast": true` / `false` | `cache_bias` 0.5 / 0.0: the router prefers experts already in memory, so replies can differ slightly |
| Thinking on by default | `"thinking": true` | the engine's own default for a request that does not choose, as `--thinking` |

A copy made on this computer (`make_bits`) counts as ready once the engine has answered one
short reply with it; that check runs after a load and its text comes back as `checked`. A load
that fails puts the memory budget back and the old model keeps serving.

The server accepts connections from this computer only, and refuses a browser
request that comes from another website (the `Origin` header) or that reaches it
under a name other than `127.0.0.1` or `localhost`. There is no CORS header, so
a web page on another origin cannot read its answers. Editor extensions call the
server from their own process and are not affected.

## When something fails

- `ECONNREFUSED` or "cannot connect": the server is not running yet (loading takes a
  while) or the address uses `localhost`; use `127.0.0.1`.
- `403` with "open this server as http://127.0.0.1:PORT": the request came with another
  host name or from a web page; use the addresses above.
- The first reply takes a long time: the prompt is read before the first token
  appears. A long chat or an agent prompt makes this longer; the next request of
  the same conversation reads only what was added.
- An agent keeps answering in plain text instead of calling a tool: the model
  does not follow the tool format well enough; try chat mode, or a larger model.
