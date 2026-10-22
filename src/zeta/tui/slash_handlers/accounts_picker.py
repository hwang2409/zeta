"""Compact named-account picker for ``/accounts``."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from ...accounts import AccountId
from .. import theme

HINT = "↑/↓ move · enter switch · d set default · n rename · esc cancel"


@dataclass
class AccountsPicker:
    accounts: tuple[AccountId, ...]
    selected: dict[str, AccountId]
    defaults: dict[str, AccountId]
    details: dict[AccountId, str] | None = None
    labels: dict[AccountId, str] | None = None
    index: int = 0

    @property
    def highlighted(self) -> AccountId:
        return self.accounts[self.index]

    def move(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.accounts)

    def render(self) -> Panel:
        rows: list[Text] = [Text("select an account", style=theme.COMMAND)]
        for position, account in enumerate(self.accounts):
            highlighted = position == self.index
            label = (self.labels or {}).get(account, account.name)
            line = Text.assemble(
                ("❯ " if highlighted else "  ", theme.ACCENT),
                (
                    f"{account.provider}:{label}",
                    theme.COMMAND if highlighted else theme.BODY,
                ),
            )
            if self.details and (detail := self.details.get(account)):
                line.append(f"  {detail}", style=theme.DIM)
            markers = []
            if self.selected.get(account.provider) == account:
                markers.append("current")
            if self.defaults.get(account.provider) == account:
                markers.append("default")
            if markers:
                line.append(f"  {', '.join(markers)}", style=theme.DIM)
            rows.append(line)
        rows.append(Text(HINT, style=theme.AFFORDANCE))
        return Panel(
            Group(*rows),
            border_style=theme.ACCENT,
            style=theme.CARD_BG,
            padding=(0, 1),
            expand=True,
        )
