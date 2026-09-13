"""
EDA — the chip-design (electronic design automation) runtime.

One package, five block types.  ``verilator``, ``yosys``, ``openroad``,
``openram`` and ``opengcram`` are separate Grafux block types but they share everything that is
expensive to build: RunPod provisioning, the SSH transport, artifact download,
the registry and the idle reaper.  So they live together here and are exposed as
five REST prefixes (``/verilator``, ``/yosys``, ``/openroad``, ``/openram``,
``/opengcram``) —
the Qt client and the orchestrator address block types by URL prefix, and the
tools' outputs genuinely differ.

They no longer share one container image.  ``image_for_kind`` in models.py picks
between the full ORFS image, the light verification image, the OpenRAM one and
the OpenGCRAM one,
because the pull dominates the wall clock of a run that itself takes seconds.

The canvas flow these blocks are built for::

    spec_hdl -> code_hdl -> verilator -> yosys -> openroad
      contract   the RTL     verify it    synth   layout + GDS

    openram -> the SRAM macro the design above instantiates
       memory parameters in; GDS, LEF, Liberty, a behavioural model and a
       SPICE netlist out.  Not a stage of the pipeline above: a source of
       one of its inputs.

    opengcram -> the same, for a GAIN-CELL memory (OpenGCRAM, an OpenRAM fork).
       Needs a technology that carries a gain cell on its tech_archive port,
       because no public one does.

Lifecycle (identical for every kind; ``{kind}`` is the tool name)::

    Regenerate -> POST /{kind}/create        (or /create_async + poll /status)
    Run        -> POST /{kind}/{id}/run      returns immediately, job runs in a thread
                  GET  /{kind}/{id}/status   poll until done
                  GET  /{kind}/{id}/result   the finished payload

Why the run is asynchronous, unlike the gpu block's synchronous ``run_gpu``: an
OpenROAD route on a real design takes 30-90 minutes.  A request that long is
killed by Render/proxies and pins a FastAPI threadpool worker for the duration,
so the run endpoint starts a job thread and the block polls.  Verilator, Yosys and a
small OpenRAM macro finish in seconds but use the same protocol on purpose — one code path here and
one in the Qt client beats a special-cased fast path.

Cost safety.  A pod bills for every second it exists, so four independent things
free it: ephemeral teardown after each run (``EDA_EPHEMERAL``), the keep-warm
deadline, the idle reaper, and the per-run watchdog (``EDA_MAX_RUN_MINUTES``).
The reaper deliberately skips records with a job in flight — it reaps on
``last_used``, which a 45-minute ``make`` would otherwise never touch.
"""
