# GPU — server-side cloud-GPU runtime for Grafux-devices.
#
# A "gpu" block represents a real, running GPU provisioned on demand from a cloud
# provider (RunPod).  Unlike the hardware device agents (Raspberry Pi, MELFA, …)
# the GPU runtime runs *inside* the devices server process and is exposed over
# REST via the FastAPI router in ``GPU.router``.
#
# The block's two actions map onto the create-once / run-many split used by the
# claw block:
#
#     Regenerate -> POST /gpu/create      -> provision (or re-provision) a pod,
#                                            returning a cached ``gpu_id``.
#     Run        -> POST /gpu/{id}/run    -> compile + execute C++/CUDA source on
#                                            the already-running pod and return the
#                                            program output plus benchmark info.
#
# A provisioned pod stays running (and billing) until it is torn down via
# DELETE /gpu/{id} (block delete / "Stop GPU") or the idle reaper in registry.py.
