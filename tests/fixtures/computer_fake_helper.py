"""A tiny stand-in for the native macOS helper, speaking its JSON-lines protocol.

Used by the helper-client tests so they exercise real process spawning, framing, timeouts, and
error mapping without a real desktop. Behavior is driven by the request op:

- ``hello``/``state``/``cursor_position``: canned ok payloads.
- ``echo``: echoes back the ``value`` arg (to check arbitrary payload fields survive framing).
- ``fail``: responds ``ok: false`` with the requested ``code``/``message``.
- ``slow``: sleeps ``seconds`` before answering (to trigger the client timeout).
- ``crash``: exits the process immediately with no response.
- ``garbage``: writes a non-JSON line.
- ``big``: returns a large payload field to force multi-chunk reads on the client side.
"""

import json
import sys
import time


def respond(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id", 0)
        op = req.get("op")
        if op == "hello":
            respond({"id": rid, "ok": True, "version": "fake", "screen_recording": False,
                     "accessibility": False})
        elif op == "state":
            respond({"id": rid, "ok": True, "locked": False, "secure_input": False,
                     "frontmost_bundle_id": "dev.zeta.fake"})
        elif op == "cursor_position":
            respond({"id": rid, "ok": True, "x": 12, "y": 34})
        elif op == "echo":
            respond({"id": rid, "ok": True, "value": req.get("value")})
        elif op == "fail":
            respond({"id": rid, "ok": False,
                     "error": {"code": req.get("code", "internal"),
                               "message": req.get("message", "boom")}})
        elif op == "slow":
            time.sleep(req.get("seconds", 5))
            respond({"id": rid, "ok": True})
        elif op == "crash":
            sys.exit(1)
        elif op == "garbage":
            sys.stdout.write("this is not json\n")
            sys.stdout.flush()
        elif op == "big":
            respond({"id": rid, "ok": True, "blob": "x" * req.get("size", 500000)})
        else:
            respond({"id": rid, "ok": False,
                     "error": {"code": "bad_request", "message": f"unknown op: {op}"}})


if __name__ == "__main__":
    main()
