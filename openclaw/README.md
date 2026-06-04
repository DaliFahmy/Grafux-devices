# OpenClaw

Server-side **claw** runtime for Grafux-devices. A *claw* is a software AI agent
assembled from a Grafux `claw` block and run against a task. Unlike the hardware
device agents (Raspberry Pi, MELFA, …), OpenClaw runs **inside the devices-server
process** and is exposed over REST — no `device_id`, no WebSocket agent.

## How a claw maps from block ports

| Claw block input port | ClawSpec field | Used as |
|-----------------------|----------------|---------|
| `soul`         | `soul`         | System prompt / persona |
| `skills`       | `skills`       | Capabilities described in the system prompt |
| `agent`        | `agent`        | Model id + params (`claude-opus-4-8` or a JSON object) |
| `credentials`  | `credentials`  | Task secrets (and an API-key fallback) |
| `api_keys`     | `api_keys`     | Anthropic API key (bare `sk-…` or JSON `{"anthropic": "…"}`) |
| `tools_config` | `tools_config` | Tool / MCP server configuration |
| `task`         | run input      | The instruction to run the claw against |
| `memory`       | run input      | Prior context prepended to the task |

Output ports: `response`, `status`, `claw_id`, `errors`.

## Endpoints (prefix `/claw`)

```
POST   /claw/create            body = ClawSpec            -> {claw_id, status}
POST   /claw/{claw_id}/run     body = {task, memory}      -> {claw_id, status, response, errors}
POST   /claw/create_and_run    body = ClawSpec            -> {claw_id, status, response, errors}
GET    /claw                                              -> [ {claw_id, name, agent} ]
GET    /claw/{claw_id}                                    -> {claw_id, name, agent}
DELETE /claw/{claw_id}                                    -> {status, claw_id}
```

The Grafux `claw` block calls `/claw/create` once (caching the returned `claw_id`
in its output port), then `/claw/{claw_id}/run` on every subsequent run.

## Configuration

- `ANTHROPIC_API_KEY` — fallback Anthropic key used when the claw's `api_keys` /
  `credentials` ports don't carry one.
- `OPENCLAW_PERSIST` — set to `1`/`true` to persist claw specs to `openclaw/_claws/`
  (off by default; the Render disk is ephemeral).

## Install & run

```bash
cd Grafux-devices
pip install -r requirements.txt          # includes anthropic
export ANTHROPIC_API_KEY=sk-...
uvicorn devices_server:app --reload
```

## Example

```bash
# 1) Provision a claw
curl -s -X POST localhost:8000/claw/create \
  -H 'Content-Type: application/json' \
  -d '{"soul":"You are a concise research assistant.",
       "skills":"web research, summarization",
       "agent":"claude-opus-4-8",
       "api_keys":"{\"anthropic\":\"sk-...\"}"}'
# -> {"claw_id":"ab12cd34ef56","status":"created"}

# 2) Run it
curl -s -X POST localhost:8000/claw/ab12cd34ef56/run \
  -H 'Content-Type: application/json' \
  -d '{"task":"Summarize the theory of relativity in 3 bullet points.","memory":""}'
# -> {"claw_id":"...","status":"ok","response":"...","errors":""}
```
