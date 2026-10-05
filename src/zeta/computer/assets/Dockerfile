# Disposable computer-use desktop for Zeta. Built by `zeta computer setup`
# inside the dedicated Lima VM only. About 1.2 GiB with Chromium.
FROM debian:bookworm-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    at-spi2-core chromium dbus-x11 fonts-dejavu-core imagemagick mousepad openbox pcmanfm \
    python3-minimal python3-pyatspi socat tint2 x11-utils x11vnc xdotool xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 65532 --user-group --home-dir /home/zeta \
        --shell /usr/sbin/nologin zeta
COPY entrypoint.sh /opt/zeta/entrypoint.sh
USER 65532:65532
ENV HOME=/home/zeta DISPLAY=:99 NO_AT_BRIDGE=0
ENTRYPOINT ["/bin/sh", "/opt/zeta/entrypoint.sh"]
