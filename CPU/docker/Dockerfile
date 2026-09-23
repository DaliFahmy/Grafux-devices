# The post-silicon verification image: a C/C++ toolchain, Python 3, and the sshd
# contract every pod in this codebase is reached through.
#
# WHY THIS IMAGE EXISTS WHEN THE TOOLCHAIN IS "JUST GCC".  Two reasons, and the
# first is not optional:
#
#   1. The SSH contract.  RunPod's own images ship openssh-server plus a start
#      script that installs the injected PUBLIC_KEY into authorized_keys.  A stock
#      ubuntu image has neither, so a pod built from one comes up RUNNING with
#      nothing listening on port 22, and wait_until_ready reports "this machine
#      does not provide direct public-IP networking" — an error that points
#      nowhere near the actual cause.  start.sh (shared verbatim with EDA) adds it.
#
#   2. Reproducible numbers.  This block exists to report durations.  A floating
#      base image means the compiler, its default flags and libc all move under
#      you, and two measurements taken a month apart stop being comparable
#      without anything visibly changing.  Pinning the tag pins the toolchain.
#
# ubuntu:22.04 matches the EDA family on purpose: the verify image is built from
# the same base, so a case timed here and a simulation timed there are running on
# comparable userland.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# One layer, because each of these is part of the same statement: what a
# verification case needs in order to be built, run and measured.
#
# NOTE the `time` package.  It installs /usr/bin/time, which is NOT the bash
# builtin of the same name — only the real binary supports `-v`, which is where
# peak RSS comes from.  Without it the benchmark silently loses max_rss_kb.
#
# `linux-tools` (perf) is deliberately NOT installed.  Hardware counters need
# CAP_PERFMON, which a RunPod container does not have; perf would install
# cleanly, run without complaint, and report zeros — a worse outcome than not
# offering the feature at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        rsync \
        ca-certificates \
        gcc \
        g++ \
        make \
        binutils \
        libc6-dev \
        python3 \
        python3-dev \
        python3-venv \
        time \
        procps \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

# The sshd contract: key-only root login, host keys generated at start.
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# The working directory EDA's runner clears before every run.  Created here so a
# run never pays for an mkdir round-trip, and so the path exists even if a run
# fails before the runner gets to it.
RUN mkdir -p /workspace/grafux

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22
CMD ["/start.sh"]
