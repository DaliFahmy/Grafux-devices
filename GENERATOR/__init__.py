"""
GENERATOR -- turn an idea (and usually a git repository) into a working block.

The app's "Generator" panel, next to Chat, works like Cursor: pick an agent
(Claude Code, or Codex -- experimental) and a model, choose Plan (propose the
block, write nothing) or Edit (build it), and describe what you want, e.g.
"a block for OpenRAM, https://github.com/VLSIDA/OpenRAM/tree/stable".

What comes out is a ``custom`` block (see ``CUSTOM/``): a manifest plus an image
in ghcr.io/dalifahmy/grafux-gen whose build ran a self-test of the tool.  Nothing
in Grafux itself is generated.

NO RUNPOD POD IS EVER RENTED HERE.  The agent runs as a GitHub Actions job and
the image is built by another; the block's first pod is the user's Regenerate --
the same rule as cpu, openram and verilator.

    agents.py    headless command lines + stream parsers for both CLIs
    contract.py  what the agent must write, how Grafux judges it, the prompts
    builder.py   GitHub Actions image builds (docker build + push to GHCR)
    actions.py   each agent turn as a GitHub Actions job (builds-repo/agent.yml)
    session.py   the conversation and the validate/build/repair loop
    router.py    /generator/*

The agent job gets the model key (the user's, else Grafux's) sealed with
GENERATOR_WRAP_KEY; the build job holds the GHCR push credential, which never
meets the agent.
"""
