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
#   docker build -t ghcr.io/<org>/grafux-eda:sky130-$(date +%Y%m%d) EDA/docker
#   docker push  ghcr.io/<org>/grafux-eda:sky130-$(date +%Y%m%d)
#
# Verify before shipping a new tag:
#   docker run --rm <img> bash -lc 'which yosys openroad klayout verilator; ls $FLOW_HOME/platforms'
#
# Local development (no RunPod account, no spend) -- this is what EDA_LOCAL_SSH=1
# in runtime.py talks to:
#   ssh-keygen -t rsa -b 2048 -f /tmp/eda_key -N ''
#   docker run -d --name grafux-eda -p 2222:22 -e PUBLIC_KEY="$(cat /tmp/eda_key.pub)" <img>
#   EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/eda_key uvicorn device.app:app

# PIN THIS.  An unpinned base changes under you and breaks provisioning silently --
# the gpu block learned that lesson with CUDA host-driver mismatches.
ARG ORFS_TAG=latest
FROM openroad/orfs:${ORFS_TAG}

# openssh-server: the transport.  rsync: faster artifact staging than raw SFTP.
# verilator: ORFS does not bundle a simulator, and the verilator block needs one.
# If the distro package is too old for the SystemVerilog the AI writes (check with
# `verilator --version`; the block wants 5.x), build it from source here instead.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        openssh-server rsync verilator \
 && mkdir -p /var/run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
 && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && rm -rf /var/lib/apt/lists/*

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22
CMD ["/start.sh"]
