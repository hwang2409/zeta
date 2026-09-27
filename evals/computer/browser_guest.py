"""Restricted browser actions for the networkless computer eval guest."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

TOOL = {
    "name": "browser",
    "description": (
        "Control one sandboxed Chromium tab in the isolated computer. Open a "
        "file:///workspace/ page, inspect its accessible snapshot, or act by "
        "exact role/name. No shell or public network is available."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "snapshot", "batch", "click", "fill", "press"],
            },
            "url": {"type": "string"},
            "role": {"type": "string"},
            "name": {"type": "string"},
            "index": {"type": "integer", "minimum": 0},
            "value": {"type": "string"},
            "key": {"type": "string"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["click", "fill", "press"],
                        },
                        "role": {"type": "string"},
                        "name": {"type": "string"},
                        "index": {"type": "integer", "minimum": 0},
                        "value": {"type": "string"},
                        "key": {"type": "string"},
                    },
                    "required": ["action", "role"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _workspace_url(url: object) -> str:
    if type(url) is not str:
        raise ValueError("open requires a file:///workspace/ URL")
    parts = urlsplit(url)
    if parts.scheme != "file" or parts.netloc or parts.query:
        raise ValueError("only file:///workspace/ URLs are allowed")
    path = Path(unquote(parts.path)).resolve()
    if not path.is_relative_to("/workspace") or not path.is_file():
        raise ValueError("page must be a regular file inside /workspace")
    return url


class BrowserGuest:
    def __init__(self) -> None:
        self.playwright: Any = None
        self.browser: Any = None
        self.page: Any = None

    def close(self) -> None:
        if self.browser is not None:
            self.browser.close()
        if self.playwright is not None:
            self.playwright.stop()

    def call(self, arguments: object) -> dict[str, Any]:
        if type(arguments) is not dict:
            raise ValueError("arguments must be an object")
        action = arguments.get("action")
        if action not in {"open", "snapshot", "batch", "click", "fill", "press"}:
            raise ValueError("unknown browser action")
        if action == "open":
            url = _workspace_url(arguments.get("url"))
        elif "url" in arguments:
            raise ValueError("url requires open action")
        if action == "batch":
            steps = arguments.get("steps")
            if type(steps) is not list or not 1 <= len(steps) <= 10:
                raise ValueError("batch requires 1 to 10 steps")
        else:
            if "steps" in arguments:
                raise ValueError("steps requires batch action")
            steps = [] if action in {"open", "snapshot"} else [arguments]
        for step in steps:
            if type(step) is not dict or step.get("action") not in {
                "click",
                "fill",
                "press",
            }:
                raise ValueError("invalid browser step")
            if type(step.get("role")) is not str or not step["role"]:
                raise ValueError("browser step requires role")
            if "name" in step and type(step["name"]) is not str:
                raise ValueError("name must be a string")
            if "index" in step and (
                type(step["index"]) is not int or step["index"] < 0
            ):
                raise ValueError("index must be nonnegative")
            field = {"fill": "value", "press": "key"}.get(step["action"])
            if field and (type(step.get(field)) is not str or (field == "key" and not step[field])):
                raise ValueError(f"{step['action']} requires {field}")

        if self.page is None:
            if action != "open":
                raise ValueError("open a page before using the browser")
            from playwright.sync_api import sync_playwright

            self.playwright = sync_playwright().start()
            # The agent cannot supply launch arguments or disable this sandbox.
            self.browser = self.playwright.chromium.launch(
                headless=True, chromium_sandbox=True
            )
            context = self.browser.new_context(
                accept_downloads=False, service_workers="block"
            )
            context.set_default_timeout(10_000)
            self.page = context.new_page()
        if action == "open":
            self.page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        for step in steps:
            locator = self.page.get_by_role(
                step["role"], name=step.get("name"), exact=True
            )
            count = locator.count()
            index = step.get("index")
            if index is None and count != 1:
                raise ValueError(
                    f"{step['action']} needs one matching element, found {count}"
                )
            if index is not None:
                if index >= count:
                    raise ValueError(
                        f"index {index} is out of range for {count} matches"
                    )
                locator = locator.nth(index)
            if step["action"] == "click":
                locator.click()
            elif step["action"] == "fill":
                locator.fill(step["value"])
            else:
                key = "Enter" if step["key"].casefold() == "enter" else step["key"]
                locator.press(key)
            self.page.wait_for_timeout(100)
        output = f"URL: {self.page.url}\nTitle: {self.page.title()}\n{self.page.aria_snapshot(mode='ai', depth=12)}"
        return {"content": [{"type": "text", "text": output[:10_000]}]}
