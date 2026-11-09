"""E8: the README's section table names exactly the dashboard pages in viz/app.py's PAGES."""

import re
from pathlib import Path

from driveflow.viz.app import PAGES

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme_pages() -> set[tuple[str, str]]:
    return set(re.findall(r"^\| ([A-Za-z& ]+) · \*\*([^*]+)\*\* \|", README.read_text(encoding="utf-8"), flags=re.M))


def test_the_readme_lists_every_page_and_no_other():
    app = {(section, label) for section, pages in PAGES.items() for _, label, _ in pages}
    readme = _readme_pages()
    assert readme, "no section table found in README.md"
    assert app - readme == set(), f"pages missing from the README: {sorted(app - readme)}"
    assert readme - app == set(), f"README pages not in PAGES: {sorted(readme - app)}"
