# Results and interpretation

The [completed notebook](../HW1_completed.ipynb) contains the full calculations, figures, and answers for Q1–Q6. The values below summarize its executed outputs. Regression uses mean squared error (MSE); classification error is the fraction of incorrectly classified observations. Lower values are better.

| Exercise | Training rows | Test rows | Features |
| --- | ---: | ---: | ---: |
| Q1: concrete, OLS and optimization | 800 | 230 | 8 |
| Q2: contaminated concrete, robust regression | 800 | 230 | 8 |
| Q3: spam classification | 2,760 | 1,841 | 57 |
| Q4: diabetes classification | 513 | 254 | 8 |
| Q5: simulated Gaussian classification | 20,000 | 50,000 | 2 |
| Q6: base-rate simulation | 1,000,000 simulated people | — | Binary test result |

These are results on the supplied splits, not cross-validation estimates. The assignment requests comparisons across hyperparameters on the test set. Reported “best” values describe those comparisons; selecting a model on the same test set would make its reported performance optimistic.

## Q1 · OLS, conditioning, and momentum

OLS achieves training MSE **101.065094** and test MSE **130.571625**. Standardizing the features using training statistics preserves both values, because centering and invertible rescaling preserve the available affine prediction functions.

The gradient-descent objective is the average squared residual divided by two. With the intercept included in design matrix A, its Hessian is H = AᵀA/n. The coefficient optimization error measures squared Euclidean distance to the OLS solution in the same feature coordinates.

| Quantity | Raw features | Standardized features |
| --- | ---: | ---: |
| Condition number of AᵀA, including intercept | 1.18385159 × 10¹⁰ | 80.3353923 |
| Strict upper stepsize bound, 2/λmax(H) | 1.19555239 × 10⁻⁶ | 0.868847575 |
| GD stepsize at half the bound | 5.97776194 × 10⁻⁷ | 0.434423788 |
| Updates to gradient norm below 10⁻⁵ | Not reached within 100,000 | 821 |

The bound is strict. At half the raw bound, GD converges in theory but makes very slow progress in low-curvature directions. At the bound, the largest-eigenvalue component alternates without decay. At twice the bound, that component grows by a factor of three per update; the implementation stops the diverging run at update 214 using a large-norm guard.

**The stopping rules matter.** The earlier raw-feature experiment follows the prescribed stepsize 6 × 10⁻⁷ and stops when the update norm falls below 10⁻⁵. It stops after **6,098 updates**, with:

| Diagnostic | Value |
| --- | ---: |
| Update norm | 9.9996061 × 10⁻⁶ |
| Gradient norm | 16.6602166 |
| Squared coefficient optimization error | 26.155452 |
| Test MSE | 131.735804 |

A small update with such a small stepsize does not establish that the gradient is small or that the coefficients are near OLS. The first iterate within 1% above the OLS test MSE occurs at update 5,579, illustrating that prediction quality can stabilize before coefficient convergence. This test-curve observation is descriptive; practical early stopping should use validation data.

On the standardized problem, all three methods use stepsize 0.434423788, zero initialization, and the same gradient-norm stopping threshold. Momentum uses β = 0.9.

| Method | Updates to gradient norm below 10⁻⁵ |
| --- | ---: |
| Gradient descent | 821 |
| Polyak momentum | 267 |
| Nesterov momentum | **147** |

Nesterov is fastest in this experiment. Momentum helps movement along persistent, low-curvature directions; the Hessian spectrum and its condition number govern the potential benefit. Gradient norms depend on the feature coordinates, so their raw and standardized magnitudes are not directly invariant under rescaling. More training data also cannot generally eliminate irreducible noise or the approximation error of a linear model.

## Q2 · Robust regression

| Model | Test MSE | Training points flagged as outliers |
| --- | ---: | ---: |
| OLS | 173.657334 | — |
| LAD, quantile 0.5 and α = 1 | **137.739939** | — |
| Huber, ε = 1.10 and α = 0 | 149.738211 | 599 |
| Huber, ε = 1.35 and α = 0 | **143.891748** | 294 |
| Huber, ε = 2.00 and α = 0 | 150.195453 | 98 |
| Huber, ε = 5.00 and α = 0 | 173.655864 | 0 |

The ranking is **LAD, best Huber, OLS**. Absolute loss reduces the influence of large residuals in the contaminated training data. LAD also has the requested L1 coefficient penalty, so this comparison does not isolate the effect of the loss alone.

Huber uses standardized features and training-only scaling statistics. Its outlier flags are based on residuals exceeding ε times an estimated residual scale; they are not known contamination labels. At ε = 5, no points are flagged and the result is approximately OLS, with a small numerical optimization difference. The behavior is not monotone across the smaller ε values.

Tukey's bounded, redescending loss can further reduce the influence of extreme residuals, but introduces nonconvex optimization, local minima, and sensitivity to initialization. Robustness to response outliers does not automatically guarantee robustness to arbitrary high-leverage feature outliers.

## Q3 · Spam classification

| Classifier | Test mistakes | Test error |
| --- | ---: | ---: |
| Logistic regression | **137 / 1,841** | **0.074416** |
| Linear discriminant analysis | 183 / 1,841 | 0.099402 |
| Gaussian Naive Bayes | 353 / 1,841 | 0.191744 |

Logistic regression uses the original feature units, L2 regularization with C = 1, and the Newton–Cholesky solver; it converges in 11 iterations. Its conditional model avoids LDA's Gaussian class-conditional assumptions. LDA can represent within-class feature correlations that Gaussian NB ignores. These are plausible explanations for this split, not a causal decomposition of the observed errors.

| Model | Nominal parameters p | Training samples per parameter, n/p |
| --- | ---: | ---: |
| LDA | 1,768 | 1.561086 |
| Gaussian NB with shared diagonal covariance, hypothetical | 172 | 16.046512 |
| GaussianNB with class-specific diagonal variances, fitted | 229 | 12.052402 |
| Logistic regression | 58 | 47.586207 |

LDA has the largest nominal parameter count, but parameter count alone does not determine test error or estimation variance. Bias, regularization, feature dependence, and sampling variation also matter. The fitted GaussianNB model generally has a quadratic boundary because it estimates separate feature variances in each class.

The five largest absolute logistic coefficients are:

| Feature | Coefficient | Odds ratio |
| --- | ---: | ---: |
| `word_freq_george` | −3.405515 | 0.033190 |
| `char_freq_%24` | 3.128006 | 22.828414 |
| `word_freq_remove` | 2.308473 | 10.059049 |
| `word_freq_000` | 1.975712 | 7.211753 |
| `word_freq_hp` | −1.827423 | 0.160828 |

Odds ratios refer to one unit in the stored feature, holding all other features fixed. They describe associations rather than causal effects. Coefficient magnitudes are not a scale-invariant feature-importance ranking, and a constant change in log odds does not imply a constant change in probability.

## Q4 · KNN and feature scaling

Logistic regression achieves test error **0.220472** (56 mistakes), below every KNN result in the requested grids.

| Feature representation and distance | Lowest-error K in its grid | Test error |
| --- | --- | ---: |
| Raw, Euclidean | 9 | 0.244094 |
| Raw, Manhattan | 15 | 0.255906 |
| Standardized, Euclidean | 41 and 81, tied | 0.240157 |
| Standardized, Manhattan | 15 | 0.228346 |

Raw KNN uses K ∈ {1, 3, 5, 9, 15}; standardized KNN additionally includes {25, 41, 61, 81, 101}. The wider standardized grid is therefore not an identical search space.

At K = 1, standardization lowers Euclidean test error from **0.374016 to 0.291339** and Manhattan test error from **0.350394 to 0.307087**. Scaling prevents the numerical units of larger-magnitude features from dominating distance.

| K | Euclidean training error, standardized | Manhattan training error, standardized |
| --- | ---: | ---: |
| 1 | 0.000000 | 0.000000 |
| 3 | 0.153996 | 0.155945 |
| 15 | 0.220273 | 0.230019 |
| 101 | 0.274854 | 0.278752 |

The zero K = 1 training error includes each point as its own neighbor. It is a resubstitution estimate, not leave-one-out validation. Small neighborhoods have high variance; large neighborhoods smooth more aggressively and may introduce bias. The finite curves do not establish a formal bias–variance decomposition, and the Euclidean minimum at K = 81 cautions against labeling every large K as underfitting. Choose K and distance with validation or cross-validation, fitting the scaler within each training fold.

## Q5 · Bayes error and model misspecification

Both supplied splits are approximately balanced: the training class counts are 10,074 and 9,926; test counts are 25,226 and 24,774. The Mahalanobis distance is **2**, giving population Bayes error **Φ(−1) = 0.1586552539**. The optimal rule predicts class 1 when X₁ − 0.8X₂ ≥ 0.6.

Each model uses the first n supplied training rows and all 50,000 test rows:

| n | LDA | Gaussian NB | Logistic regression |
| ---: | ---: | ---: | ---: |
| 100 | 0.16404 | 0.27762 | 0.16374 |
| 200 | 0.16482 | 0.27092 | 0.16424 |
| 500 | 0.15898 | 0.27338 | 0.15986 |
| 1,000 | 0.15968 | 0.28056 | 0.16026 |
| 2,000 | 0.15846 | 0.26338 | 0.15794 |
| 5,000 | 0.15794 | 0.27034 | 0.15812 |
| 10,000 | 0.15802 | 0.27116 | 0.15798 |
| 20,000 | 0.15816 | 0.27478 | 0.15826 |

LDA and logistic regression approach the Bayes rate because the common-covariance Gaussian model and the linear posterior log odds are correctly specified here. Gaussian NB ignores conditional correlation 0.8; its limiting rule uses X₁ alone and has error Φ(−0.6), approximately **0.274253**. More observations reduce estimation error but cannot remove this misspecification.

The Bayes lower bound concerns population risk, not every empirical test error. The known Bayes rule has error **0.158220** on the supplied finite test set. At n = 20,000, LDA is approximately **0.000495 below** the population rate, only **0.30** approximate test-error standard errors. The standard error at the Bayes rate is **0.001634**, so the small gap does not demonstrate improvement on the population optimum. Nested subsets and a shared test set also mean that the plotted estimates are not independent replications.

## Q6 · Base rates and repeated evidence

With prevalence 0.1%, sensitivity 99%, and specificity 99%:

| Quantity | Exact result |
| --- | ---: |
| P(disease given a positive result) | 0.0901639344, or 9.0164% |
| P(disease given a negative result) | 0.0000101110, or 0.00101110% |
| Positive likelihood ratio | 99 |
| Prevalence yielding posterior at least 50% after one positive | 1% |
| Posterior after two conditionally independent positives | 0.9075 |
| Minimum independent positives for posterior above 99% | 3 |

The fixed-seed simulation of one million people produces:

| Disease status | Positive | Negative |
| --- | ---: | ---: |
| Disease | 958 | 9 |
| No disease | 9,938 | 989,095 |

Its positive predictive value is **958 / 10,896 = 0.08792217**, consistent with the exact probability within Monte Carlo variation.

Under equal-cost classification error and using only this binary result, the Bayes rule always predicts no disease because both posterior probabilities are below 0.5. Its population error is **0.001**, versus **0.010** for trusting the test. Their simulated errors are **0.000967** and **0.009947**, respectively. The positive result remains informative: it changes the posterior from 0.1% to about 9%, and can matter under different decision costs or for further information gathering.

Three independent positive results yield a posterior of approximately **99.8971%**. Multiplying likelihood ratios requires conditional independence within each disease-status group. Shared measurement errors or persistent individual factors can invalidate that assumption, just as correlated features invalidate Gaussian NB's product model in Q5. These conclusions concern the stated probability model.

## Reading the comparisons

Training-only standardization prevents preprocessing leakage. It does not make repeated test-set comparisons into valid hyperparameter selection. Differences on a single split should be read as empirical observations, and small changes across systems may reflect floating-point or solver tolerances. See [Reproducibility](REPRODUCIBILITY.md) for the environment and verification workflow.
