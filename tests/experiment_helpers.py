import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def materialized_experiment_inputs(config):
    """Materializa entradas mínimas para testes unitários de manifesto."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for key in ("catalog", "dataset", "feedback"):
            path = root / config["paths"][key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture-{key}\n", encoding="utf-8")
        yield root
