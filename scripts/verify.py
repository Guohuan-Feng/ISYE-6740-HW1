#!/usr/bin/env python3
"""Check the saved HW1 notebook, or execute a fresh copy into build/.

Run from any directory with the repository's requirements installed:
    python scripts/verify.py
    python scripts/verify.py --execute

Fresh execution uses this Python interpreter, not an unrelated installed kernel.
The tracked notebook and gallery are never overwritten.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "HW1_completed.ipynb"
DATASETS = tuple(
    f"Q{question}_{split}.{'xls' if question <= 2 else 'csv'}"
    for question in range(1, 6)
    for split in ("train", "test")
)
FIGURES = (
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
PROBE_MARKER = "HW1_VERIFICATION_JSON:"

# This final, temporary cell checks numerical invariants without requiring exact
# optimizer iteration counts or byte-identical plots across operating systems.
# It is removed before the executed notebook is saved.
NUMERICAL_PROBE = r'''
import json as _hw1_json

def _hw1_finite(name, values, lower=None, upper=None):
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise AssertionError(f"{name}: missing or nonfinite numerical results")
    if lower is not None and np.any(array < lower):
        raise AssertionError(f"{name}: values below {lower}")
    if upper is not None and np.any(array > upper):
        raise AssertionError(f"{name}: values above {upper}")

_hw1_mses = {
    "Q1 OLS training MSE": q1_train_mse,
    "Q1 OLS testing MSE": q1_test_mse,
    "Q1 standardized OLS training MSE": q1_std_train_mse,
    "Q1 standardized OLS testing MSE": q1_std_test_mse,
    "Q2 OLS testing MSE": q2_ols_mse,
    "Q2 LAD testing MSE": q2_lad_mse,
    "Q2 standardized OLS testing MSE": q2_ols_scaled_mse,
}
_hw1_finite("regression MSEs", list(_hw1_mses.values()), lower=0)
_hw1_finite("Huber MSEs", q2_huber_results["testing MSE"], lower=0)
if not np.allclose([q1_train_mse, q1_test_mse],
                   [q1_std_train_mse, q1_std_test_mse], rtol=1e-7, atol=1e-7):
    raise AssertionError("Q1 OLS predictions changed after invertible scaling")
if not np.isclose(q2_ols_mse, q2_ols_scaled_mse, rtol=1e-7, atol=1e-7):
    raise AssertionError("Q2 OLS predictions changed after invertible scaling")

_hw1_errors = {
    "Q3 LDA testing error": q3_lda_error,
    "Q3 Gaussian NB testing error": q3_gnb_error,
    "Q3 logistic testing error": q3_lr_error,
    "Q4 logistic testing error": q4_lr_error,
    "Q5 theoretical Bayes error": q5_bayes_rate,
    "Q5 empirical Bayes error": q5_empirical_bayes,
}
_hw1_finite("classification errors", list(_hw1_errors.values()), 0, 1)
for _hw1_label, _hw1_values in [
    ("raw Euclidean KNN", q4_raw_euclidean["Test error"]),
    ("raw Manhattan KNN", q4_raw_manhattan["Test error"]),
    ("standardized KNN", q4_std_results["Test error"]),
    ("KNN training errors", q4_training_results["Training error"]),
    ("Q5 learning curves", q5_curve.to_numpy()),
]:
    _hw1_finite(_hw1_label, _hw1_values, 0, 1)

_hw1_momentum_iterations = {}
for _hw1_method, _hw1_run in q1_momentum_runs.items():
    _hw1_finite(f"{_hw1_method} optimization errors", _hw1_run["opt_error"], 0)
    _hw1_finite(f"{_hw1_method} gradient norms", _hw1_run["grad_norm"], 0)
    if _hw1_run["grad_norm"][-1] >= 1e-5:
        raise AssertionError(f"{_hw1_method} did not meet its gradient tolerance")
    _hw1_momentum_iterations[_hw1_method] = int(_hw1_run["iterations"])
# Deliberately unstable Q1(h) runs are not required to converge.

_hw1_finite("Q6 posteriors", [q6_p_d_positive, q6_p_d_negative,
                             q6_empirical_positive, q6_empirical_negative], 0, 1)
_hw1_counts = [q6_tp, q6_fn, q6_fp, q6_tn]
if sum(_hw1_counts) != q6_population_size or min(_hw1_counts) < 0:
    raise AssertionError("Q6 four-way counts do not partition the population")
if not np.isclose(q6_p_d_positive, q6_p_from_odds):
    raise AssertionError("Q6 probability and odds forms disagree")

_hw1_report = {
    "checks": ["finite nonnegative regression errors", "OLS scaling invariance",
               "classification errors in [0, 1]", "standardized optimization convergence",
               "Q6 valid probabilities, population counts, and odds identity"],
    "metrics": {key: float(value) for key, value in {**_hw1_mses, **_hw1_errors}.items()},
    "momentum_iterations": _hw1_momentum_iterations,
    "simulation_population": int(q6_population_size),
}
print("HW1_VERIFICATION_JSON:" + _hw1_json.dumps(_hw1_report, allow_nan=False))
'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def check_data() -> dict:
    """Check all supplied datasets and, when present, their recorded hashes."""
    for filename in DATASETS:
        path = ROOT / "data" / filename
        require(path.is_file() and path.stat().st_size > 0,
                f"Missing or empty dataset: data/{filename}")

    manifest_path = ROOT / "data" / "manifest.json"
    checked = 0
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files")
        require(isinstance(entries, list), "Manifest must contain a 'files' list")
        expected_paths = {f"data/{name}" for name in DATASETS}
        listed_paths = [entry["path"] for entry in entries]
        require(len(listed_paths) == len(set(listed_paths)), "Duplicate manifest file paths")
        require(set(listed_paths) == expected_paths,
                "Manifest must describe exactly the ten supplied datasets")
        for entry in entries:
            content = (ROOT / entry["path"]).read_bytes()
            require(sha256(content) == entry["sha256"],
                    f"Dataset hash mismatch: {entry['path']}")
            if "bytes" in entry:
                require(len(content) == entry["bytes"],
                        f"Dataset byte count mismatch: {entry['path']}")
            checked += 1
    return {"files_present": len(DATASETS), "manifest_hashes_checked": checked}


def check_notebook(notebook, *, compare_gallery: bool) -> dict:
    nbformat.validate(notebook)
    require(len(notebook.cells) == 83, "Expected 83 assignment notebook cells")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(len(code_cells) == 37, "Expected 37 completed code cells")
    counts = [cell.execution_count for cell in code_cells]
    require(counts == list(range(1, len(code_cells) + 1)),
            "Code cells must have execution counts 1 through 37 in order")
    images = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        require(bool(cell.source.strip()), f"Empty code cell {index}")
        for output in cell.outputs:
            require(output.output_type != "error",
                    f"Notebook error output in cell {index}: {output.get('ename', '')}")
            encoded = output.get("data", {}).get("image/png")
            if encoded is not None:
                if isinstance(encoded, list):
                    encoded = "".join(encoded)
                png = base64.b64decode("".join(encoded.split()), validate=True)
                require(png.startswith(b"\x89PNG\r\n\x1a\n"),
                        f"Invalid PNG output in cell {index}")
                images.append(png)
    require(len(images) == len(FIGURES), "Expected nine embedded PNG figures")
    image_hashes = {}
    for filename, image in zip(FIGURES, images):
        digest = sha256(image)
        if compare_gallery:
            gallery_path = ROOT / "figures" / filename
            require(gallery_path.is_file(), f"Missing gallery figure: figures/{filename}")
            require(sha256(gallery_path.read_bytes()) == digest,
                    f"Gallery figure differs from saved notebook output: {filename}")
        image_hashes[filename] = digest
    return {"cells": len(notebook.cells), "executed_code_cells": len(code_cells),
            "png_outputs": len(images), "gallery_hashes_checked": compare_gallery,
            "figure_sha256": image_hashes}


def execute_notebook(notebook, build_dir: Path, timeout: int) -> tuple:
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    # A temporary kernelspec makes execution portable even when python3 points
    # to a different virtual environment or has not been registered at all.
    with tempfile.TemporaryDirectory(prefix="hw1-verification-") as temporary:
        kernels_dir = Path(temporary) / "kernels"
        spec_dir = kernels_dir / "hw1-verification"
        spec_dir.mkdir(parents=True)
        spec = {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "HW1 verification", "language": "python",
            "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        }
        (spec_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
        specs = KernelSpecManager(kernel_dirs=[str(kernels_dir)], ensure_native_kernel=False)
        manager = AsyncKernelManager(kernel_name="hw1-verification", kernel_spec_manager=specs)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        probe = nbformat.v4.new_code_cell(NUMERICAL_PROBE)
        notebook.cells.append(probe)
        client = NotebookClient(
            notebook, km=manager, timeout=timeout, allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute(cleanup_kc=True)
            texts = "".join(output.get("text", "") for output in probe.outputs
                            if output.output_type == "stream")
            payloads = [line[len(PROBE_MARKER):] for line in texts.splitlines()
                        if line.startswith(PROBE_MARKER)]
            require(len(payloads) == 1, "Fresh execution did not produce its numerical report")
            numerical_report = json.loads(payloads[0])
        finally:
            notebook.cells.pop()
            # Keep a failed run for diagnosis too; the original file is untouched.
            nbformat.write(notebook, build_dir / NOTEBOOK.name)
    return notebook, numerical_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute a fresh copy and write build/ artifacts")
    parser.add_argument("--timeout", type=int, default=600, help="execution timeout per cell in seconds (default: 600)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    started = time.monotonic()
    build_dir = ROOT / "build"
    report = {"status": "failed", "mode": "execute" if args.execute else "saved",
              "python": sys.version.split()[0]}
    if args.execute:
        build_dir.mkdir(exist_ok=True)
    try:
        report["datasets"] = check_data()
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        report["saved_notebook"] = check_notebook(notebook, compare_gallery=True)
        print("Saved notebook passed: 83 cells, 37 ordered executions, 9 matching gallery PNGs.", flush=True)
        print(f"Datasets passed: {report['datasets']['files_present']} files; "
              f"{report['datasets']['manifest_hashes_checked']} manifest hashes checked.", flush=True)
        if args.execute:
            print(f"Executing a fresh notebook with {sys.executable} ...", flush=True)
            notebook, report["numerical_checks"] = execute_notebook(notebook, build_dir, args.timeout)
            report["fresh_notebook"] = check_notebook(notebook, compare_gallery=False)
            from nbconvert import HTMLExporter
            html, _ = HTMLExporter().from_notebook_node(notebook)
            (build_dir / "HW1_completed.html").write_text(html, encoding="utf-8")
            print("Fresh execution and numerical checks passed.", flush=True)
            print("Saved build/HW1_completed.ipynb, build/HW1_completed.html, and build/validation.json.")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"Verification failed: {error}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if args.execute:
            (build_dir / "validation.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
