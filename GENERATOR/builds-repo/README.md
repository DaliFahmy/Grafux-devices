# grafux-gen-builds — image builds for Generator blocks

The devices server's Generator (`GENERATOR/builder.py`) commits each generated
block's build context to a branch `build/<tag>` of THIS repository and dispatches
`.github/workflows/build.yml`, which runs `docker build` (the Dockerfile ends with
Grafux's self-test) and pushes `ghcr.io/dalifahmy/grafux-gen:<tag>`.

## One-time setup

1. Create a repository, e.g. `DaliFahmy/grafux-gen-builds`, and copy
   `.github/workflows/build.yml` from here onto its `main` branch.
2. Package `ghcr.io/dalifahmy/grafux-gen` (created by Grafux-devices'
   "Build custom example image" workflow):
   - make it **Public** (RunPod pulls anonymously);
   - Package settings → *Manage Actions access* → add `grafux-gen-builds` with
     **Write**, so this repo's `GITHUB_TOKEN` can push to it. (Alternatively add
     a `GHCR_PUSH_TOKEN` secret holding a PAT with `write:packages`.)
3. Create a fine-grained token for the devices server scoped to THIS repository
   only, with **Contents: read & write** and **Actions: read & write**.
4. On the devices server (render.yaml / dashboard):
   - `GENERATOR_GITHUB_TOKEN` = that token
   - `GENERATOR_BUILDS_REPO` = `DaliFahmy/grafux-gen-builds`
   - optionally `GENERATOR_BUILD_WORKFLOW` (default `build.yml`),
     `GENERATOR_BUILDS_REF` (default `main`).

## Why a separate repository

Build branches contain code the agent wrote and that the Dockerfile's `RUN`
steps execute. Keeping them out of the Grafux repos means a generated build can
never touch Grafux's own workflows or secrets, and the server's token is scoped
to nothing else.

Public vs private: in a public repo builds are free and every generated build
context is public. Generated images are public anyway (see above), so v1 assumes
public upstream projects only.
