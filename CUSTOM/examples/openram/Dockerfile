# The hand-written reference custom block: OpenRAM behind the manifest contract.
#
# It reuses the published native image (OpenRAM + its venv + sshd/start.sh) and
# adds only what the contract needs: /opt/grafux/run.sh + its adapter, and a
# self-test as the LAST step, so an image that cannot compile its own smoke
# macro fails to build instead of failing on a user's rented pod.
FROM ghcr.io/dalifahmy/grafux-openram:v1_2_48-scn4m-20260912

COPY run.sh adapter.py selftest.sh /opt/grafux/
COPY grafux-block.json /opt/grafux/grafux-block.json
RUN chmod +x /opt/grafux/run.sh /opt/grafux/selftest.sh

RUN /opt/grafux/selftest.sh

EXPOSE 22
CMD ["/start.sh"]
