import pytest

def pytest_collect_file(parent, path):
    if path.ext == ".py" and path.basename.startswith("test"):
        return pytest.Module.from_parent(parent, fspath=path)
