"""
CUSTOM -- blocks defined by a manifest instead of by code.

One block type, ``custom``, at the REST prefix ``/custom``.  Every native
pod-backed block (openram, cpu, ...) is Python here plus C++ in the app; a custom
block is ONE JSON manifest (``manifest.py`` is the spec) plus the image it names.
The Generator (Claude Code / Codex in a sandbox pod) writes both, so users can
bring their own projects and open-source tools into Grafux without anyone
editing Grafux.

The split that keeps that safe: Grafux's side -- this runner, the app's EDA
executor, the port writer -- is written once and never generated.  Everything
generated lives INSIDE the image (a Dockerfile, a ``run.sh`` adapter, a
self-test).  A bad generation yields a broken block, never a broken Grafux.

Like ``CPU/``, it registers with EDA rather than forking its lifecycle:

    EDA.models.register_kind_image("custom", ...)    disk (no default image, on purpose)
    EDA.runtime.register_runner("custom", run_custom)

Lifecycle (the EDA contract, unchanged)::

    Regenerate -> POST /custom/create_async  (image = manifest.runtime.image)
    Run        -> POST /custom/{id}/run      {manifest, inputs:{port: text}}
                  GET  /custom/{id}/status   poll until done
                  GET  /custom/{id}/result   outputs keyed by the manifest's ports
"""
