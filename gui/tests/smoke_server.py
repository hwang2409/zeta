"""Deterministic real-protocol server for the opt-in native smoke driver."""

import argparse
import asyncio
import os
from pathlib import Path

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.server import ZetaServer
from zeta.types import TextContent, ToolCall


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    # Require isolation explicitly; never fall back to the user's real home.
    home = Path(os.environ["ZETA_HOME"])
    turns = [
        ScriptedTurn(
            content=[TextContent("I will inspect the project before changing it.")],
            tool_calls=[ToolCall("demo-read", "bash", {"command": "pwd"})],
        ),
        ScriptedTurn(
            content=[
                TextContent(
                    "## Core chat loop\n\n"
                    "The native interface includes:\n\n"
                    "- Readable session previews\n"
                    "- Streaming markdown and tool receipts\n"
                    "- A multiline composer with keyboard controls\n\n"
                    '```rust\nfn main() {\n    println!("Hello from zeta");\n}\n```'
                )
            ],
            usage={"input_tokens": 128, "output_tokens": 96},
        ),
    ]
    server = ZetaServer(
        home=home,
        socket_path=args.socket,
        provider="fake",
        backend_factory=lambda p, m, h: (FakeBackend(turns), m or "offline"),
    )
    await server.start()
    print("ready", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
