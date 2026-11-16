"""A tiny stand-in for the native macOS helper, speaking its JSON-lines protocol.

Used by the helper-client tests so they exercise real process spawning, framing, timeouts, and
error mapping without a real desktop. Behavior is driven by the request op:

- ``hello``/``state``/``cursor_position``: canned ok payloads.
- ``screenshot``: a 1x1 PNG frame; ``display: 99`` is a sentinel that returns a ``secure_input``
  error so the MacBackend error mapping can be tested. Input ops (``move``/``click``/...) ok.
- ``echo``: echoes back the ``value`` arg (to check arbitrary payload fields survive framing).
- ``fail``: responds ``ok: false`` with the requested ``code``/``message``.
- ``slow``: sleeps ``seconds`` before answering (to trigger the client timeout).
- ``crash``: exits the process immediately with no response.
- ``garbage``: writes a non-JSON line.
- ``big``: returns a large payload field to force multi-chunk reads on the client side.
"""

import base64
import json
import sys

# a complete 1x1 PNG so MacBackend.screenshot decodes real image bytes.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


def respond(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    killed = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id", 0)
        op = req.get("op")
        if killed and op != "hello":
            respond({"id": rid, "ok": False,
                     "error": {"code": "killed", "message": "kill switch engaged"}})
        elif op == "kill":
            killed = True
            respond({"id": rid, "ok": True})
        elif op == "hello":
            respond({"id": rid, "ok": True, "version": "fake", "screen_recording": False,
                     "accessibility": False})
        elif op == "state":
            respond({"id": rid, "ok": True, "locked": False, "secure_input": False,
                     "frontmost_bundle_id": "dev.zeta.fake"})
        elif op == "cursor_position":
            respond({"id": rid, "ok": True, "x": 12, "y": 34})
        elif op == "screenshot":
            # display 99 is a sentinel so MacBackend error-mapping can be tested.
            if req.get("display") == 99:
                respond({"id": rid, "ok": False,
                         "error": {"code": "secure_input", "message": "password field"}})
            else:
                respond({"id": rid, "ok": True, "png_b64": base64.b64encode(_PNG_1x1).decode(),
                         "width": 1, "height": 1, "scale": 2.0, "origin_x": 0, "origin_y": 0})
        elif op in ("move", "click", "mouse_down", "mouse_up", "drag", "type", "key", "scroll"):
            respond({"id": rid, "ok": True})
        elif op == "echo":
            respond({"id": rid, "ok": True, "value": req.get("value")})
        elif op == "fail":
            respond({"id": rid, "ok": False,
                     "error": {"code": req.get("code", "internal"),
                               "message": req.get("message", "boom")}})
        elif op == "block":
            # Wait for another line; the client must terminate this process on deadline.
            sys.stdin.readline()
        elif op == "crash":
            sys.exit(1)
        elif op == "garbage":
            sys.stdout.write("this is not json\n")
            sys.stdout.flush()
        elif op == "partial":
            sys.stdout.write('{"id":' + str(rid))
            sys.stdout.flush()
            return
        elif op == "wrong_id":
            respond({"id": rid + 1, "ok": True})
        elif op == "bad_shape":
            respond({"id": rid, "ok": "yes"})
        elif op == "bad_error":
            respond({"id": rid, "ok": False, "error": "bad"})
        elif op == "big":
            respond({"id": rid, "ok": True, "blob": "x" * req.get("size", 500000)})
        else:
            respond({"id": rid, "ok": False,
                     "error": {"code": "bad_request", "message": f"unknown op: {op}"}})


if __name__ == "__main__":
    main()
