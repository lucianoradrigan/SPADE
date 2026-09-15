# Empty on purpose -- its only job is to exist here. pytest adds every directory that contains a
# conftest.py to sys.path early in collection, so this guarantees the repo root (and therefore
# `experiments/`, imported by tests/test_registry.py and tests/test_train_model.py) is importable
# regardless of how pytest is invoked. `python -m pytest` already puts the cwd on sys.path, which
# is why this bug never showed up locally -- CI's bare `pytest -v` doesn't get that for free.
