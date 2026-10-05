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
