# The Generator sandbox: where Claude Code / Codex read a repo and write a block.
#
# It builds NOTHING itself (images are built by GitHub Actions, see
# GENERATOR/builder.py) -- it needs git to clone, node for the two CLIs, python3
# and common build tools so the agent can explore a project's build system, and
# the sshd + PUBLIC_KEY start.sh every Grafux pod is reached through.
#
# Pin the CLIs: an agent CLI that changes its JSONL schema under us turns every
# session into a parse error.  Bump deliberately, re-run the parser tests against
# captured output, then publish a new tag and move GENERATOR_IMAGE.
FROM ubuntu:22.04

ARG CLAUDE_CODE_VERSION=latest
ARG CODEX_VERSION=latest
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server ca-certificates curl git gnupg \
        python3 python3-pip python3-venv build-essential jq ripgrep unzip \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && claude --version && codex --version

# Headless: no telemetry prompts, no auto-updates mid-session.
RUN printf '%s\n' \
      'export DISABLE_AUTOUPDATER=1' \
      'export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1' \
      'export PATH=/usr/local/bin:$PATH' \
      > /etc/profile.d/grafux-generator.sh

RUN mkdir -p /workspace/src /workspace/gen /workspace/.grafux
WORKDIR /workspace

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22
CMD ["/start.sh"]
