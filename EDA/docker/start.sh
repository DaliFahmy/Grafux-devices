#!/bin/sh
# Reproduces the RunPod PUBLIC_KEY contract that stock openroad/orfs lacks:
# install the injected public key, generate host keys, then hand the container
# over to sshd in the foreground (so the container's life is the daemon's life).
set -e

if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
else
    # Loud on purpose: with no key the pod comes up healthy but unreachable, and
    # the resulting timeout blames the network rather than the missing env var.
    echo "WARNING: PUBLIC_KEY is not set - no one will be able to SSH into this pod." >&2
fi

# Host keys are absent in a fresh image; without them sshd refuses to start.
ssh-keygen -A

exec /usr/sbin/sshd -D -e
