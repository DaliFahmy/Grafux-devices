# grafux-gen-builds — where the Generator's agent and image builds run

The devices server's Generator never rents a RunPod pod (only a block's
**Regenerate** does). Instead, it uses two workflows in THIS repository:

| Workflow | Dispatched by | What it does |
|---|---|---|
| `.github/workflows/agent.yml` | `GENERATOR/actions.py` | One agent **turn**: installs the pinned Claude Code / Codex CLI, clones the upstream repo, runs the agent on branch `session/<id>`, streams its output to the devices server, commits the block's files back. |
| `.github/workflows/build.yml` | `GENERATOR/builder.py` | Builds the block's image from branch `build/<tag>` (the Dockerfile ends with Grafux's self-test) and pushes `ghcr.io/dalifahmy/grafux-gen:<tag>`. |

## One-time setup

1. Create the repository, **private** recommended, since the `session/*`
   branches contain users' prompts and generated code:
   `gh repo create DaliFahmy/grafux-gen-builds --private`. Put both workflow
   files from here on its `main` branch.
2. **`GENERATOR_WRAP_KEY`** is a Fernet key that seals the model key and the
   callback token passed to the agent job (workflow inputs are readable by
   anyone who can read the repo). Generate one:
   `python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"`
   - set it as this repo's secret: `gh secret set GENERATOR_WRAP_KEY -R DaliFahmy/grafux-gen-builds`
   - set the **same** value as `GENERATOR_WRAP_KEY` on the devices service (Render).
3. Package `ghcr.io/dalifahmy/grafux-gen`: Package settings, then *Manage Actions
   access*, then add `grafux-gen-builds` with **Write**, so `build.yml` can
   push. (Or add a `GHCR_PUSH_TOKEN` secret holding a PAT with `write:packages`.)
   **Required:** without it every build passes its self-test and then fails at
   *Push* with `denied: permission_denied: write_package`. The package was created by
   another repo's workflow, so this repo's `GITHUB_TOKEN` starts with no access to it.
   The package must stay **Public** (RunPod pulls anonymously).
4. Create a fine-grained token scoped to THIS repository only, with
   **Contents: read & write** and **Actions: read & write**. On the devices
   service set:
   - `GENERATOR_GITHUB_TOKEN` = that token
   - `GENERATOR_BUILDS_REPO` = `DaliFahmy/grafux-gen-builds`
   - optionally `GENERATOR_PUBLIC_URL` (defaults to Render's `RENDER_EXTERNAL_URL`),
     which is where the agent job streams live output.

## Security notes

- The agent step never holds this repo's `GITHUB_TOKEN`: checkout uses
  `persist-credentials: false`, and only the final "Save the turn" step gets it.
- The model key is unsealed into `/workspace/.grafux/keys.env` for the agent
  step and deleted before anything is committed. It is still readable by the
  upstream repo's code while the agent runs, so use a dedicated, spend-limited
  key for `GENERATOR_ANTHROPIC_API_KEY` / `GENERATOR_OPENAI_API_KEY`.
- The image build (`build.yml`) holds the GHCR push credential; `docker build`
  RUN steps do not see it.
