"""Składanie elementów programu w całość."""
from dataclasses import dataclass

from .describer import BackgroundDescriber
from .gpu import GpuArbiter
from .store import Store
from .vectors import VectorIndex

DB_FILE = "index.sqlite"
LOG_FILE = "filesearcher.log"


@dataclass
class App:
    config: object
    store: Store
    index: VectorIndex
    embedder: object
    arbiter: GpuArbiter
    background: BackgroundDescriber | None


def ensure_embedder(store: Store, embedder, ask) -> bool:
    """Wektory z różnych modeli są nieporównywalne – przy zmianie modelu baza idzie od nowa."""
    stored = store.get_meta("embedder_id")
    if stored is not None and stored != embedder.model_id:
        question = (
            f"Baza została zbudowana modelem {stored}, a teraz wybrany jest {embedder.model_id}.\n"
            "Przebudować bazę od nowa? [t/N] "
        )
        if not ask(question):
            return False
        store.clear_all()
    store.set_meta("embedder_id", embedder.model_id)
    store.set_meta("dim", str(embedder.dim))
    return True


def open_app(config, embedder, describer, ask, device: str = "cpu") -> App | None:
    store = Store(config.baza_dir / DB_FILE)
    if not ensure_embedder(store, embedder, ask):
        store.close()
        return None
    index = VectorIndex(embedder.dim, device=device)
    ids, kinds, mat = store.all_vectors()
    if len(ids):
        index.add(ids, kinds, mat)
    arbiter = GpuArbiter()
    background = None
    if describer is not None and config.opisy:
        background = BackgroundDescriber(store, index, embedder, describer, arbiter, config.baza_dir / LOG_FILE)
    return App(config, store, index, embedder, arbiter, background)
