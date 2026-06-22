# device.agents — code that runs *on* the hardware.
#
# Each sub-package is a self-contained agent for one kind of device.  An agent
# connects out to the devices hub over a WebSocket and executes the command
# blocks it receives (compile/run/shell/robot motion/…) locally:
#
#     raspberry_pi  — generic Linux SBC agent (download-from-S3, compile, run)
#     windows       — Windows workstation agent (adds screenshot / camera capture)
#     melfa         — Mitsubishi MELFA robot agent (ROS 2 / MoveIt2 bridge)
#     jetson        — NVIDIA Jetson agent (placeholder)
#     arduino       — Arduino agent (placeholder)
