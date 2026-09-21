# ISYE 6740 — HW1

Completed Assignment 1: regression, optimization, robust regression, linear classification, KNN, and Bayes error/probabilities.

## Submission

Upload **HW1_completed.ipynb** to Canvas. It contains all answers, 37 executed code cells, and nine figures. `HW1_completed.html` is a reading copy. Fill in the optional teammate name if applicable.

## Reproduce

Use Python 3.12. Download or clone this repository and keep the ten data files alongside the notebook.

```bash
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace HW1_completed.ipynb
```

Alternatively, open the notebook in Jupyter or VS Code and run all cells from top to bottom. Training and testing files are used as supplied; preprocessing is fitted using training statistics only. The simulation uses a fixed random seed (6740). Tested package versions are pinned in `requirements.txt`.
