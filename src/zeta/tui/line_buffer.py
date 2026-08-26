"""Small buffer for streamed text lines."""


class LineBuffer:
    def __init__(self) -> None:
        self.value = ""

    def feed(self, value: str) -> list[str]:
        self.value += value
        lines = self.value.split("\n")
        self.value = lines.pop()
        return lines

    def flush(self) -> list[str]:
        if not self.value:
            return []
        line = self.value
        self.value = ""
        return [line]
