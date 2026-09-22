# Assignment datasets

These ten files come from the **HW1.zip supplied by the user** for ISYE 6740 Assignment 1. The original training/testing split, row order, values, and column order are retained. Q6 uses a simulated population and has no data file.

| Question | Dataset | Training rows | Testing rows | Features | Target (last column) |
|---|---|---:|---:|---:|---|
| Q1 | Concrete compressive strength | 800 | 230 | 8 | `Concrete compressive strength(MPa, megapascals)` |
| Q2 | Concrete data with contaminated training observations | 800 | 230 | 8 | `Concrete compressive strength(MPa, megapascals)` |
| Q3 | Spam email classification | 2,760 | 1,841 | 57 | `class`: 1 = spam, 0 = not spam |
| Q4 | Diabetes classification | 513 | 254 | 8 | `Outcome`: 1 = diabetes, 0 = no diabetes |
| Q5 | Simulated two-class Gaussian data | 20,000 | 50,000 | 2 | `Label`: 0 or 1 |

Row counts exclude the header. Each filename identifies its question and split, such as `Q1_train.xls` and `Q1_test.xls`.

## Schemas

- **Q1 and Q2:** the first eight columns are cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. The last column is concrete compressive strength in MPa. The first seven features are material quantities in kg per cubic meter; age is in days. The workbook headers contain some whitespace, including a trailing space in the target header, which is preserved in the files and recorded exactly in the manifest.
- **Q3:** the first 57 columns contain email features: word frequencies, character frequencies, and capital-run statistics. The final column contains the binary spam label. Exact original feature names are recorded in the manifest.
- **Q4:** features are `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`, in that order. The ninth column is `Outcome`.
- **Q5:** columns are `X1`, `X2`, and `Label`. The assignment specifies equal class priors, means `(0, 0)` and `(1.2, 0)`, and shared covariance `[[1, 0.8], [0.8, 1]]`.

## Loading and verification

Read the `.xls` files with `pandas.read_excel(..., engine="xlrd")` and CSV files with `pandas.read_csv(...)`. In every file, use all columns except the last as features and the final column as the response. Fit any scaling transformation on training features only.

[manifest.json](manifest.json) lists each repository-relative path, byte count, SHA256 digest, dimensions, feature count, and exact column names. Its hashes cover the repository files as stored. Dataset bytes match the extracted source archive, including its original line endings.

This directory documents assignment provenance and does not assert an additional license for the supplied datasets.
