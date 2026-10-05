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
