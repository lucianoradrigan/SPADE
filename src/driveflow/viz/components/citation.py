"""SPADE's authors and how to cite the platform (overview and Documentation). The README repeats
CITATION_TEXT and every author; tests/test_ui_contract.py keeps them in step."""

from html import escape

import streamlit as st

#: (name, e-mail, affiliation), in citation order.
AUTHORS = [
    ("Luciano Radrigan", "lradrigan@udec.cl", "Electrical Engineering Department, Universidad de Concepcion, Chile"),
    ("Anibal S. Morales", "animorales@udec.cl", "Electrical Engineering Department, Universidad de Concepcion, Chile"),
    ("Felipe Ruiz", "felipe.ruiz@uss.cl", "Energy Transition Center, University San Sebastian, Santiago, Chile"),
    ("Jose Rodriguez", "jose.rodriguezp@uss.cl", "Centro de Transicion Energetica (CTE), Facultad de Ingenieria, Universidad San Sebastian, Chile"),
]

#: Public profiles by author name, as (label, URL).
PROFILES = {
    "Luciano Radrigan": [("LinkedIn", "https://www.linkedin.com/in/lucianoradrigan"), ("Google Scholar", "https://scholar.google.com/citations?user=ynRKFTIAAAAJ&hl=es&oi=ao")],
}

CITATION_TEXT = ('L. Radrigan, A. S. Morales, F. Ruiz, and J. Rodriguez, "SPADE: Simulation Platform for Analysis, Diagnosis & Electric drives," '
                 "version 0.1.0, 2026. [Online]. Available: https://github.com/lucianoradrigan/SPADE")

CITATION_BIBTEX = """@misc{spade2026,
  author = {Radrigan, Luciano and Morales, Anibal S. and Ruiz, Felipe and Rodriguez, Jose},
  title  = {{SPADE}: Simulation Platform for Analysis, Diagnosis \\& Electric drives},
  year   = {2026},
  note   = {Version 0.1.0},
  url    = {https://github.com/lucianoradrigan/SPADE}
}"""


def _profile_links(name: str) -> str:
    links = " · ".join(f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>' for label, url in PROFILES.get(name, []))
    return f'<p class="spade-author-links">{links}</p>' if links else ""


def render_citation() -> None:
    st.markdown("##### :material/format_quote: Authors & how to cite")
    st.markdown("SPADE was created by the authors below. **If you use SPADE in your work, please cite them.**")
    cards = "".join(f'<div><b>{escape(name)}</b><span>{escape(place)}</span><a href="mailto:{escape(mail)}">{escape(mail)}</a>{_profile_links(name)}</div>' for name, mail, place in AUTHORS)
    st.html(f'<div class="spade-root spade-authors">{cards}</div>')
    text, bibtex = st.tabs(["Text", "BibTeX"])
    with text:
        st.code(CITATION_TEXT, language=None, wrap_lines=True)
    with bibtex:
        st.code(CITATION_BIBTEX, language="bibtex")
