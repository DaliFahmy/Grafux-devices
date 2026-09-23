"""
CPU — the post-silicon verification runtime.

One block type, ``cpu``: compile a verification case, run it on a real CPU, and
report both the verdict and how long it took.  Exposed at the REST prefix
``/cpu``, because the Qt client and the orchestrator address block types by URL.

WHAT THIS BLOCK IS FOR.  ``spec_hdl -> code_hdl -> testbench -> verilator``
closes the loop on a design *before* tape-out.  This one runs *after*: the chip
exists, it cannot be changed, and the question is whether the silicon in front of
you actually behaves — and how fast.  The case it runs is normally written by a
``post_silicon_verification`` block, which generates the source and says what the
case proves; this package only compiles it, runs it and measures it::

    post_silicon_verification -> cpu
      code       -> code        the program to run
      language   -> language    c | cpp | python
      verifies   -> spec        what the case CLAIMS to prove, wire-only evidence
      feedback   <- analysis    the return leg, after a run

WHY IT REUSES ``EDA/``.  Provisioning a RunPod pod, the SSH transport, artifact
download, the registry, the idle reaper and all four cost-safety mechanisms are
identical to what the EDA tools need and are genuinely expensive to get right.
So this package owns only what is actually CPU-specific — the build/run matrix,
the benchmark, and the verdict — and registers itself with EDA for the rest:

    EDA.models.register_kind_image("cpu", ...)   the image and disk this kind wants
    EDA.runtime.register_runner("cpu", run_cpu)  what to do once SSH is up

It deliberately does NOT fork ``_start_job``/``_run_job``.  That is where the
orphan-pod handling, the keep-warm-versus-ephemeral precedence and the run
watchdog live; a second copy would drift, and drift there bills real money.

Lifecycle (the EDA contract, unchanged)::

    Regenerate -> POST /cpu/create          (or /create_async + poll /status)
    Run        -> POST /cpu/{id}/run        returns immediately, job runs in a thread
                  GET  /cpu/{id}/status     poll until done
                  GET  /cpu/{id}/result     the finished payload

The run is asynchronous for the reason ``EDA/__init__.py`` gives, and it applies
here with particular force: a benchmark is a warmup plus N measured repetitions,
so even a case that runs in a second takes many to measure honestly.

ON THE NUMBERS.  Timing is taken ON THE POD with a nanosecond clock, so network
latency is excluded by construction, and the case is run ``repetitions`` times
after ``warmup`` discarded runs — a single sample on a shared cloud vCPU is
noise, not a measurement.  ``machine`` reports what it ran on, on its own port
rather than buried inside ``benchmark``, because on a CPU benchmark the machine
is the one caveat that makes every cross-run comparison meaningful or worthless.
``perf`` counters are deliberately absent: a container without CAP_PERFMON
returns zeros for them rather than failing, which is worse than not offering them.
"""
