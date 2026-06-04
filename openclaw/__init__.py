# OpenClaw — server-side claw runtime for Grafux-devices.
#
# A "claw" is a software AI agent assembled from a small set of inputs
# (soul, skills, agent, credentials, api_keys, tools_config) and run against a
# task.  Unlike the hardware device agents (Raspberry Pi, MELFA, …) the claw
# runtime runs *inside* the devices server process and is exposed over REST via
# the FastAPI router in ``openclaw.router``.
