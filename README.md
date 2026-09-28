# 🏏 WPL Win Probability & Match Analytics Engine

A machine learning-powered cricket analytics system that estimates **win probability during Women's Premier League (WPL) matches** and reconstructs how the predicted probability changes throughout an innings.

The project combines historical WPL ball-by-ball data, feature engineering, machine learning, and an interactive Streamlit dashboard.

---

## 🎯 Project Objective

The objective of this project is to estimate the probability of the chasing team winning a WPL match based on the current match state.

The model uses information such as:

- Runs scored
- Runs required
- Balls remaining
- Wickets in hand
- Current run rate
- Required run rate
- Recent scoring momentum
- Match phase

The system can also replay historical WPL matches delivery by delivery and generate a model-based win-probability progression.

---

## 📊 Dataset

The project uses **Women's Premier League ball-by-ball data from Cricsheet**.

The processed dataset contains:

- **88 WPL matches**
- **20,656 deliveries**
- **23 original delivery-level fields**

The data covers WPL seasons from:

- 2023
- 2024
- 2025
- 2026

One tied match decided by a Super Over was excluded from binary win/loss model training.

---

## 🧹 Data Processing

The raw match JSON files are flattened into a delivery-level dataset.

Important processing steps include:

1. Parsing match metadata
2. Extracting innings and delivery information
3. Identifying legal deliveries
4. Calculating cumulative runs
5. Calculating wickets lost
6. Calculating wickets in hand
7. Calculating runs required
8. Calculating balls remaining
9. Calculating current run rate
10. Calculating required run rate
11. Calculating recent scoring momentum
12. Assigning the match phase

### Match Phases

| Phase | Overs |
|---|---|
| Powerplay | 0–5 |
| Middle | 6–14 |
| Death | 15–19 |

---

## 🧠 Machine Learning

The primary model is **Logistic Regression**.

### Input Features

#### Numerical Features

- `runs_scored`
- `runs_required`
- `balls_remaining`
- `wickets_in_hand`
- `current_run_rate`
- `required_run_rate`
- `runs_last_12_balls`
- `runs_last_6_balls`

#### Categorical Feature

- `phase`

The categorical phase feature is encoded using one-hot encoding.

---

## 🔬 Model Validation

A **match-level train/test split** was used instead of randomly splitting individual deliveries.

This prevents deliveries from the same match appearing in both training and testing data.

### Dataset Split

- Training matches: **69**
- Testing matches: **18**
- Total modeling matches: **87**

### Final Logistic Regression Performance

| Metric | Result |
|---|---:|
| Accuracy | **90.98%** |
| ROC-AUC | **96.66%** |
| Log Loss | **0.2520** |
| Brier Score | **0.0720** |

The final model was selected based on its overall performance, particularly its probability-quality metrics.

---

## 🤖 Model Comparison

An XGBoost model and calibrated Logistic Regression model were also evaluated.

| Model | Accuracy | ROC-AUC | Log Loss | Brier Score |
|---|---:|---:|---:|---:|
| Logistic Regression | **90.98%** | 96.66% | **0.2520** | **0.0720** |
| Calibrated Logistic Regression | **90.98%** | **96.86%** | 0.2600 | 0.0732 |
| XGBoost | 86.55% | 94.68% | 0.2967 | 0.0942 |

The original Logistic Regression model is used as the final prediction model.

---

## 📈 Historical Match Replay

The project includes a match replay engine that processes the second innings of historical WPL matches delivery by delivery.

For every legal delivery, the system:

```text
Current delivery
       ↓
Updated match state
       ↓
Feature calculation
       ↓
Logistic Regression
       ↓
Win probability
       ↓
Next delivery
       ↓
Updated probability