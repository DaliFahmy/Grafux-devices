# The Grafux EDA image: OpenROAD-flow-scripts plus Verilator, reachable over SSH.
#
# WHY THIS IMAGE EXISTS AT ALL.  Every pod in this codebase is driven over SSH with
# a key injected through RunPod's PUBLIC_KEY env var.  RunPod's own images ship
# openssh-server and a start script that installs that key into authorized_keys;
# upstream openroad/orfs ships neither, and RunPod does NOT add them for arbitrary
# images -- it only sets the env var.  A pod built straight from openroad/orfs
# therefore comes up RUNNING with nothing listening on port 22, wait_until_ready
# never finds an endpoint, and the user sees "this machine does not provide direct
# public-IP networking" -- an error that points at the network and is really about
# a missing daemon.  These few lines are what prevent that.
#
# Build and push (pin the tag; see EDA/models.py DEFAULT_IMAGE):
#   docker build -t ghcr.io/dalifahmy/grafux-eda:sky130-20260901 --build-arg ORFS_TAG=26Q3-422-ga7de7c42f EDA/docker
#   docker push  ghcr.io/dalifahmy/grafux-eda:sky130-20260901
#
# The pushed GHCR package must be PUBLIC: RunPod pulls anonymously, and a
# private package fails the pull silently -- the pod comes up with no image and
# times out with the same misleading 'no public-IP networking' error as a
# missing sshd.
#
# Verify before shipping a new tag (note FLOW_HOME is NOT exported by the base
# image -- _EDA_ENV in pod_client.py supplies it, so set it here too):
#   docker run --rm <img> bash -lc 'export FLOW_HOME=/OpenROAD-flow-scripts/flow; \
#     command -v yosys openroad klayout verilator sshd; verilator --version; \
#     ls $FLOW_HOME/platforms'
# Expect: verilator 5.x, and sky130hd among the platforms.
#
# Local development (no RunPod account, no spend) -- this is what EDA_LOCAL_SSH=1
# in runtime.py talks to:
#   ssh-keygen -t rsa -b 2048 -f /tmp/eda_key -N ''
#   docker run -d --name grafux-eda -p 2222:22 -e PUBLIC_KEY="$(cat /tmp/eda_key.pub)" <img>
#   EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/eda_key uvicorn device.app:app

# PIN THIS.  An unpinned base changes under you and breaks provisioning silently --
# the gpu block learned that lesson with CUDA host-driver mismatches.
ARG ORFS_TAG=26Q3-422-ga7de7c42f
FROM openroad/orfs:${ORFS_TAG}

# openssh-server: the transport.  rsync: faster artifact staging than raw SFTP.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        openssh-server rsync \
 && mkdir -p /var/run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
 && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && rm -rf /var/lib/apt/lists/*

# Verilator, built from source rather than installed from apt.
#
# ORFS does not bundle a simulator, and Ubuntu 22.04's package is Verilator
# *4.038* — released in 2020, with SystemVerilog support far behind what an LLM
# writing RTL will happily emit. A design that simulates fine everywhere else
# would fail here, and the error would look like the user's fault rather than the
# toolchain's. 5.x is the line worth holding.
#
# The base image already carries every build dependency (git, make, g++,
# autoconf, flex, bison, perl, help2man, FlexLexer.h, zlib) because it builds
# OpenROAD itself, so this needs no extra apt packages — just the compile, which
# is the bulk of this image's build time.
#
# `make install` lands it in /usr/local/bin, already on PATH, so _EDA_ENV in
# pod_client.py needs no change.
ARG VERILATOR_TAG=v5.050
RUN git clone --depth 1 --branch "${VERILATOR_TAG}" \
        https://github.com/verilator/verilator.git /tmp/verilator \
 && cd /tmp/verilator \
 && autoconf \
 && ./configure \
 && make -j"$(nproc)" \
 && make install \
 && cd / \
 && rm -rf /tmp/verilator

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22
CMD ["/start.sh"]
