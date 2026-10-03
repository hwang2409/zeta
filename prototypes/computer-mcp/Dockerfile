FROM debian:bookworm-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    dbus-x11 fonts-dejavu-core imagemagick mousepad openbox x11-utils x11vnc \
    xdotool xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 65532 --user-group --home-dir /home/zeta \
        --shell /usr/sbin/nologin zeta
COPY entrypoint.sh /opt/zeta/entrypoint.sh
USER 65532:65532
ENV HOME=/home/zeta DISPLAY=:99
ENTRYPOINT ["/bin/sh", "/opt/zeta/entrypoint.sh"]
