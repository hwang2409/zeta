FROM python:3.12-alpine

COPY guest.py /opt/zeta/guest.py

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["sleep", "1800"]
