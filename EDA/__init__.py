"""
EDA — the chip-design (electronic design automation) runtime.

One package, three block types.  ``verilator``, ``yosys`` and ``openroad`` are
separate Grafux block types but they share everything that is expensive to build:
the container image, the PDK, RunPod provisioning, the SSH transport, artifact
download, the registry and the idle reaper.  So they live together here and are
exposed as three REST prefixes (``/verilator``, ``/yosys``, ``/openroad``) — the
Qt client and the orchestrator address block types by URL prefix, and the three
tools' outputs genuinely differ.

The canvas flow these blocks are built for::

    code (language=verilog) -> verilator -> yosys -> openroad
       describe the chip       verify it    synth   layout + GDS

Lifecycle (identical for all three kinds; ``{kind}`` is the tool name)::

    Regenerate -> POST /{kind}/create        (or /create_async + poll /status)
    Run        -> POST /{kind}/{id}/run      returns immediately, job runs in a thread
                  GET  /{kind}/{id}/status   poll until done
                  GET  /{kind}/{id}/result   the finished payload

Why the run is asynchronous, unlike the gpu block's synchronous ``run_gpu``: an
OpenROAD route on a real design takes 30-90 minutes.  A request that long is
killed by Render/proxies and pins a FastAPI threadpool worker for the duration,
so the run endpoint starts a job thread and the block polls.  Verilator and Yosys
finish in seconds but use the same protocol on purpose — one code path here and
one in the Qt client beats a special-cased fast path.

Cost safety.  A pod bills for every second it exists, so four independent things
free it: ephemeral teardown after each run (``EDA_EPHEMERAL``), the keep-warm
deadline, the idle reaper, and the per-run watchdog (``EDA_MAX_RUN_MINUTES``).
The reaper deliberately skips records with a job in flight — it reaps on
``last_used``, which a 45-minute ``make`` would otherwise never touch.
"""
