# Credit Risk Probability Model for Alternative Data

> **Bati Bank × eCommerce Partner** — An end-to-end implementation for building, deploying, and automating a credit risk model using alternative transactional data.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
3. [Data Description](#data-description)
4. [Project Structure](#project-structure)
5. [Methodology](#methodology)
6. [Getting Started](#getting-started)

---

## Project Overview

Bati Bank is partnering with an eCommerce platform to offer a **buy-now-pay-later** service. Customers who qualify receive credit to purchase products, with repayment scheduled over a defined period. This project delivers the underlying credit scoring infrastructure: a model that ingests raw transaction data, engineers behavioral signals, estimates a probability of default, and exposes the result via a containerized REST API.

**Key deliverables:**

- A proxy target variable for default, engineered from RFM (Recency, Frequency, Monetary) behavioral segmentation
- A reproducible feature engineering pipeline
- Trained, tuned, and MLflow-tracked classification models
- A credit score derived from predicted risk probability
- A containerized REST API with CI/CD automation

---

## Credit Scoring Business Understanding

### 1. Basel II and the Imperative for Interpretable, Well-Documented Models

The Basel II Capital Accord, and its successor frameworks, fundamentally reshaped how banks quantify, document, and justify credit risk. Under the **Internal Ratings-Based (IRB) approach**, institutions are permitted to use their own internal models to estimate key risk parameters — primarily Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) — as direct inputs to regulatory capital calculations. However, this permission comes at a price: banks must satisfy rigorous conditions around model **transparency, validation, and supervisory disclosure** before and after approval.

This has direct implications for model design:

- **Interpretability is not optional.** Regulators require that a bank be able to explain how its model assigns risk weights and that those assignments are consistent with observed default rates over time. A black-box model that cannot be interrogated by a risk officer, auditor, or regulator fails this standard regardless of its predictive accuracy on held-out data.

- **Documentation is a first-class artifact.** Basel II's Pillar 2 (Supervisory Review) and Pillar 3 (Market Disclosure) collectively require that banks maintain comprehensive model documentation covering data sources, variable selection rationale, validation methodology, and ongoing performance monitoring. This project therefore treats its methodology write-up — including proxy variable justification, feature engineering decisions, and model comparison results — as a regulatory-grade artifact, not an afterthought.

- **Model governance and monitoring are ongoing obligations.** Once deployed, a model must be monitored for population stability and discriminatory power. If the underlying data distribution shifts (e.g., changes in eCommerce customer behavior), the model must be recalibrated or replaced — and that process must itself be documented.

In short, Basel II transforms the credit model from a pure predictive tool into a **regulated, auditable decision system**. Every modeling choice must be defensible in plain language to a non-technical supervisor.

**Implication for this project:** To align with Basel II principles, all feature engineering steps, RFM target construction procedures, model training experiments, evaluation metrics, and deployment decisions will be documented, version controlled, and reproducible. MLflow tracking, Git history, and automated testing provide the audit trail expected in a regulated credit-risk environment.

---

### 2. Why a Proxy Variable Is Necessary — and the Business Risks It Introduces

The raw transaction dataset from the eCommerce partner contains **no direct default label**. There is no field indicating whether a customer failed to repay a prior loan, became delinquent, or was written off. This is a common challenge in alternative credit scoring contexts: behavioral data is rich in signal, but it was collected for purposes (fraud detection, merchandising) other than credit underwriting.

**Why a proxy is necessary:**

Supervised classification models require a target variable. Without a label for "default" or "good payer," there is nothing to train against. The solution is to engineer a **proxy target** — a variable derived from observable behavior that is strongly correlated with actual creditworthiness. In this project, Recency, Frequency, and Monetary (RFM) metrics are calculated for every customer and standardized before clustering. K-Means segmentation with three customer segments is then used to identify customer groups, and the least engaged cluster is labeled as `is_high_risk = 1`. Customers who transact infrequently, have not transacted recently, and spend little are hypothesized to exhibit behavioral patterns consistent with higher credit risk, making this label the binary training target.

This approach draws directly on established practice in alternative credit scoring. As the HKMA's white paper on alternative credit scoring notes, transactional data — including cashflow patterns and purchasing behavior — can be used to construct creditworthiness assessments when traditional credit bureau data is unavailable. RFM-derived signals fall squarely within this category of alternative data.

**Business risks introduced by proxy-based prediction:**

Using a proxy instead of a real default label introduces several risks that must be acknowledged and managed:

| Risk                            | Description                                                                                                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Label noise**                 | The proxy may misclassify customers — a low-frequency buyer might be a cautious, creditworthy saver, not a risk. Any error in the proxy propagates directly into model training.                                                                                                |
| **Construct validity**          | The proxy measures _engagement with the eCommerce platform_, not _willingness and ability to repay debt_. These are related but not identical constructs. A customer could be highly active on the platform yet overextended on credit elsewhere.                               |
| **Regulatory scrutiny**         | Regulators may question whether the proxy variable truly represents default risk under Basel II definitions. The methodology must be thoroughly documented and validated against any available ground-truth data (e.g., subsequent loan performance if the product is piloted). |
| **Feedback loops**              | If the model trained on the proxy drives loan approvals, and loan outcomes are then used to refine the proxy, a biased feedback loop can emerge — systematically disadvantaging certain customer segments without anyone recognizing the circular logic.                        |
| **Fairness and discrimination** | Behavioral proxies can inadvertently encode demographic or socioeconomic patterns. Customers from lower-income segments may have lower RFM scores for reasons unrelated to creditworthiness, leading to discriminatory outcomes.                                                |

Mitigating these risks requires: clear documentation of proxy construction assumptions, sensitivity analysis on proxy thresholds, ongoing comparison of model predictions against actual loan performance once the product launches, and regular fairness audits across customer segments.

---

### 3. Trade-offs Between Interpretable and High-Performance Models in a Regulated Context

One of the central tensions in credit modeling is the trade-off between **predictive performance** and **regulatory acceptability**. This is not a generic ML trade-off; it is specifically acute in banking because the stakes of model decisions are high, the regulatory environment is explicit, and adverse outcomes (wrongful denial of credit, unanticipated losses) have measurable legal and financial consequences.

#### Simple, Interpretable Models — Logistic Regression with Weight of Evidence (WoE)

The classical approach to credit scorecard development uses logistic regression on features transformed via **Weight of Evidence (WoE)** binning. This methodology has been the industry standard for decades and remains widely used.

**Advantages:**

- **Fully interpretable:** Each feature's contribution to the final score is a linear, additive term. A credit officer can trace exactly why a customer received a particular score.
- **Regulatory alignment:** WoE + logistic regression naturally produces a scorecard format that regulators understand, auditors can review, and risk committees can approve.
- **Monotonic relationships enforced:** WoE binning allows analysts to enforce business-logical constraints (e.g., higher income should always reduce risk score), preventing counterintuitive model behavior.
- **Stable and robust:** Logistic regression generalizes well and is less prone to overfitting on small or noisy datasets common in alternative data contexts.
- **Adverse action explanations:** Regulatory frameworks in many jurisdictions require that rejected applicants receive a reason for their denial. Logistic regression scorecards make this trivial; each top negative factor corresponds directly to a model coefficient.

**Disadvantages:**

- **Limited expressiveness:** Linear models cannot capture complex interactions between features (e.g., a customer who spends heavily _and_ transacts very infrequently may be a very different risk profile than the additive model assumes).
- **Manual feature engineering burden:** WoE binning requires significant analyst judgment and iterative effort. Automated approaches can introduce their own biases if not carefully supervised.

#### High-Performance Models — Gradient Boosting

Ensemble tree methods consistently outperform logistic regression on tabular credit data, particularly when features have non-linear relationships and complex interactions.

**Advantages:**

- **Superior discrimination:** Gradient boosting typically achieves higher AUC-ROC and Gini coefficients, meaning it better separates good and bad borrowers in the population.
- **Handles raw features:** Tree models are robust to outliers, missing values, and non-monotonic relationships without extensive preprocessing.
- **Captures interactions automatically:** The model can learn that the combination of low recency + high value + low frequency signals a dormant high-spender — a nuanced pattern a linear model would miss.

**Disadvantages:**

- **Interpretability deficit:** A gradient boosting ensemble with hundreds of trees has no simple, human-readable form. While tools like SHAP (SHapley Additive exPlanations) can provide post-hoc explanations, these approximations may not satisfy regulators who require intrinsic interpretability.
- **Adverse action complexity:** Generating compliant denial reason codes from a SHAP explanation is possible but operationally harder to audit and defend than a scorecard coefficient.
- **Overfitting risk on proxy labels:** The higher capacity of gradient boosting means it can overfit to the noise in a proxy label, learning spurious patterns that reflect the proxy construction methodology rather than true credit risk.
- **Model governance overhead:** Complex models require more extensive validation infrastructure, more frequent recalibration checks, and higher technical expertise to maintain — all of which increase operational cost and regulatory risk.

#### Recommendation for This Project

Given that Bati Bank operates in a regulated financial context and is building a _new_ credit product with _proxy-derived_ labels, the recommended approach is a **hybrid strategy**:

1. **Primary production model:** Logistic Regression with WoE-transformed features, fully documented as a regulatory-grade scorecard. This model is deployable, explainable, and auditable from day one.
2. **Challenger models:** Decision Tree, Random Forest, and Gradient Boosting tracked in MLflow as performance benchmarks. If a challenger substantially outperforms the scorecard and can be paired with SHAP-based explanations that satisfy internal model risk management, it may be promoted to production in a future iteration — after appropriate validation and regulatory consultation.

This phased approach prioritizes compliance and trust-building in the early stages of the product while preserving the option to adopt more powerful methods as the model matures and real loan performance data becomes available.

---

## Data Description

The dataset contains transaction-level records from the Xente eCommerce platform. Each row represents a single customer transaction.

| Field                  | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `TransactionId`        | Unique transaction identifier                             |
| `BatchId`              | Batch processing identifier                               |
| `AccountId`            | Unique customer account identifier                        |
| `SubscriptionId`       | Customer subscription identifier                          |
| `CustomerId`           | Customer identifier                                       |
| `CurrencyCode`         | Transaction currency                                      |
| `CountryCode`          | Geographical country code                                 |
| `ProviderId`           | Source provider of item purchased                         |
| `ProductId`            | Product/item name                                         |
| `ProductCategory`      | Broader product category                                  |
| `ChannelId`            | Purchase channel (web, Android, iOS, pay-later, checkout) |
| `Amount`               | Transaction value; positive = debit, negative = credit    |
| `Value`                | Absolute value of Amount                                  |
| `TransactionStartTime` | Transaction timestamp                                     |
| `PricingStrategy`      | Xente merchant pricing category                           |
| `FraudResult`          | Fraud flag: 1 = fraud, 0 = not fraud                      |

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### Step 1 — Proxy Target Variable Engineering

RFM (Recency, Frequency, Monetary) features are computed per customer from the transaction log and standardized. K-Means clustering with three customer segments partitions customers by engagement level. The least engaged cluster is assigned `is_high_risk = 1`, becoming the binary training label.

### Step 2 — Feature Engineering Pipeline

Raw transactions are aggregated to customer level. Features include transaction counts, mean/total spend, product diversity, channel preference, and time-based patterns. WoE binning is applied for the logistic regression scorecard.

### Step 3 — Model Training and Experiment Tracking

Four classifiers are trained (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) with hyperparameter tuning. All experiments are tracked in MLflow with metrics (AUC-ROC, Gini, KS statistic), parameters, and artifacts logged.

### Step 4 — Credit Score Derivation

The best model's predicted probability of default is scaled to a credit score in the range 300–850 using a standard Points-to-Double-Odds (PDO) transformation.

### Step 5 — API Deployment

The selected model is serialized and served via a FastAPI endpoint. The service accepts a customer feature payload and returns a risk probability, credit score, and risk tier. The container is built and tested automatically through a GitHub Actions CI/CD pipeline.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tsi122124/credit-risk-model.git
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Run data processing and feature engineering
python src/data_processing.py

# Train models (logs to MLflow)
python src/train.py

# Start the API locally
uvicorn src.api.main:app --reload

# Build and run with Docker
docker-compose up --build
```

## References

1. Basel Committee on Banking Supervision. International Convergence of Capital Measurement and Capital Standards (Basel II).
2. Hong Kong Monetary Authority (HKMA). Alternative Credit Scoring of MSMEs.
3. World Bank Group. Credit Scoring Approaches Guidelines.
4. Credit Scoring Statistical Analysis (Statistica Sinica).
5. Corporate Finance Institute. Credit Risk Overview.
