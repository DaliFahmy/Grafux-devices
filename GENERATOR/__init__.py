"""
GENERATOR -- turn an idea (and usually a git repository) into a working block.

The app's "Generator" panel, next to Chat, works like Cursor: pick an agent
(Claude Code, or Codex -- experimental) and a model, choose Plan (propose the
block, write nothing) or Edit (build it), and describe what you want, e.g.
"a block for OpenRAM, https://github.com/VLSIDA/OpenRAM/tree/stable".

What comes out is a ``custom`` block (see ``CUSTOM/``): a manifest plus an image
in ghcr.io/dalifahmy/grafux-gen, verified by a self-test at build time and by a
real run on a fresh pod.  Nothing in Grafux itself is generated.

    agents.py    headless command lines + stream parsers for both CLIs
    contract.py  what the agent must write, how Grafux judges it, the prompts
    builder.py   GitHub Actions image builds (docker build + push to GHCR)
    session.py   the sandbox pod, the turn, and the validate/build/smoke/repair loop
    router.py    /generator/*

The agent runs in a RunPod pod from ``docker/Dockerfile`` with the user's own
model key; the build runs in GitHub Actions (``builds-repo/``), where the push
credential never meets the repo's code.
"""
