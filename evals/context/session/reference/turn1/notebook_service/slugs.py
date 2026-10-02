import re
import unicodedata


def normalize(title: str) -> str:
    text = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    )
    slug = "-".join(re.findall(r"[a-z0-9]+", text))
    if not slug or slug == "root":
        raise ValueError("invalid slug")
    return slug
