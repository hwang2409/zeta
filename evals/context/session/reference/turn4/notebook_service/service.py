import json

from .models import Note
from .slugs import normalize
from .store import Store


class Notebook:
    def __init__(self):
        self.store = Store()

    def create(self, title, body):
        slug = normalize(title)
        if slug in self.store.notes:
            raise ValueError("duplicate slug")
        note = Note(slug, title, body)
        self.store.notes[slug] = note
        return note

    def export(self):
        return "".join(
            json.dumps(
                {"slug": n.slug, "title": n.title, "body": n.body},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for n in sorted(self.store.notes.values(), key=lambda x: x.slug)
        )

    @classmethod
    def load(cls, text):
        book = cls()
        try:
            rows = text.splitlines()
            for line in rows:
                value = json.loads(line)
                if list(value) != ["slug", "title", "body"] or not all(
                    isinstance(v, str) for v in value.values()
                ):
                    raise ValueError("invalid record")
                if (
                    normalize(value["title"]) != value["slug"]
                    or value["slug"] in book.store.notes
                ):
                    raise ValueError("invalid slug")
                book.store.notes[value["slug"]] = Note(
                    value["slug"], value["title"], value["body"]
                )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid data") from exc
        return book
