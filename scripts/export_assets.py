"""Regenerate published reading copies and PNGs from saved notebook outputs."""

from __future__ import annotations

import base64
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = (
    "Q1e_GD_optimization_error.png",
    "Q1f_GD_testing_MSE.png",
    "Q1h_stepsize_comparison.png",
    "Q1j_momentum_comparison.png",
    "Q4a_Euclidean_KNN.png",
    "Q4b_Manhattan_KNN.png",
    "Q4d_standardized_KNN.png",
    "Q5a_training_scatter.png",
    "Q5c_learning_curves.png",
)


def main() -> None:
    notebook = nbformat.read(ROOT / "HW1_completed.ipynb", as_version=4)
    nbformat.validate(notebook)
    images = [
        output["data"]["image/png"]
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    if len(images) != len(FIGURE_NAMES):
        raise ValueError(f"Expected {len(FIGURE_NAMES)} figures, found {len(images)}")
    if any(
        output.get("output_type") == "error"
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    ):
        raise ValueError("Notebook contains an error output; fix it before export")
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    for name, encoded in zip(FIGURE_NAMES, images, strict=True):
        (figures / name).write_bytes(base64.b64decode(encoded))

    exporter = HTMLExporter(template_name="lab")
    report, _ = exporter.from_notebook_node(notebook)
    (ROOT / "HW1_completed.html").write_text(report, encoding="utf-8")
    navigation = '''
<nav aria-label="Report navigation" style="font-family:system-ui,sans-serif;
padding:14px 24px;background:#10243a;color:white;display:flex;gap:24px;
flex-wrap:wrap;align-items:center">
<strong>ISYE 6740 · HW1</strong>
<a style="color:#d9edff" href="https://github.com/Guohuan-Feng/ISYE-6740-HW1">Repository &amp; figure gallery</a>
<a style="color:#d9edff" href="https://github.com/Guohuan-Feng/ISYE-6740-HW1/blob/main/HW1_completed.ipynb">Submission notebook</a>
<a style="color:#d9edff" href="https://github.com/Guohuan-Feng/ISYE-6740-HW1/blob/main/docs/RESULTS.md">Results summary</a>
</nav>
'''
    body_start = report.find("<body")
    body_end = report.find(">", body_start) + 1
    if body_start < 0 or body_end == 0:
        raise ValueError("HTML exporter did not produce a body element")
    published = report[:body_end] + navigation + report[body_end:]
    published = published.replace("<title>Notebook</title>", "<title>ISYE 6740 · HW1 — Complete Report</title>")
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(published, encoding="utf-8")
    (docs / ".nojekyll").touch()
    print(f"Exported {len(images)} figures, offline HTML, and docs/index.html")


if __name__ == "__main__":
    main()
