"""Guard against the ReadTheDocs mock list silently going stale.

The Sphinx build (``docs/conf.py``) does not install bbstrader's third-party
dependencies; it mocks them via ``autodoc_mock_imports``. If a module imports a
package that is neither installed for the docs build nor mocked, autodoc fails
to import that module and renders an *empty* section. Worse, because each
package ``__init__`` star-imports its submodules, one missing dependency blanks
the entire package on the rendered site.

This test fails loudly the moment a new top-level third-party import is added
without updating the mock list, so the breakage is caught in CI instead of on
the published docs.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "bbstrader"
CONF = REPO_ROOT / "docs" / "conf.py"

# Packages the docs build installs for real (see docs/requirements.txt); these
# do not need to be mocked.
INSTALLED_FOR_DOCS = frozenset({"numpy", "pandas", "pytz"})

# Imported lazily inside a function body (not at module top level), so its
# absence never breaks an autodoc import. catalog.py imports pyarrow this way
# and falls back to CSV when it is missing.
LAZY_OPTIONAL = frozenset({"pyarrow"})


def _top_level_third_party_imports() -> dict[str, set[str]]:
    """Map each third-party top-level import name to the files that use it."""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name and name != "bbstrader" and name not in stdlib:
                    found.setdefault(name, set()).add(path.relative_to(SRC).as_posix())
    return found


def _mock_imports() -> set[str]:
    """Parse ``autodoc_mock_imports`` from conf.py without importing it.

    Importing conf.py has side effects (it rewrites ``sys.path`` and injects a
    fake C++ extension module), so the literal list is read statically instead.
    """
    tree = ast.parse(CONF.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "autodoc_mock_imports"
            for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("autodoc_mock_imports not found in docs/conf.py")


def test_every_third_party_import_is_covered():
    imports = _top_level_third_party_imports()
    covered = INSTALLED_FOR_DOCS | LAZY_OPTIONAL | _mock_imports()
    uncovered = {name: files for name, files in imports.items() if name not in covered}
    assert not uncovered, (
        "These third-party imports are neither installed for the docs build, "
        "mocked in docs/conf.py, nor known lazy imports, so they will blank "
        "their package on ReadTheDocs. Add them to autodoc_mock_imports "
        f"(use the import name, not the distribution name): {uncovered}"
    )
