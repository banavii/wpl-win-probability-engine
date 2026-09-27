# ============================================================
# WPL CRICKET WIN PROBABILITY & MATCH ANALYTICS ENGINE
# ============================================================
#
# Complete pipeline:
#
# 1. Load processed Cricsheet data
# 2. Feature engineering
# 3. Match-rule validation
# 4. Create win target
# 5. Remove invalid post-win states
# 6. Match-level train/test split
# 7. Logistic Regression
# 8. XGBoost comparison
# 9. Calibration analysis
# 10. Calibrated Logistic Regression
# 11. Model comparison
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.calibration import (
    calibration_curve,
    CalibratedClassifierCV
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_parquet(
    "data/processed/deliveries.parquet"
)

print(
    "Dataset shape:",
    df.shape
)

print("\nColumns:")

print(
    df.columns.tolist()
)


# ============================================================
# 3. SORT DATA
# ============================================================

df = df.sort_values(
    [
        "match_id",
        "innings",
        "over",
        "ball"
    ]
).reset_index(drop=True)


# ============================================================
# 4. LEGAL DELIVERY
# ============================================================

print("\n" + "=" * 60)
print("CREATING LEGAL DELIVERY FEATURE")
print("=" * 60)

# Wides and no-balls do not count as legal deliveries.

df["legal_delivery"] = (
    ~df["extra_type"].isin(
        ["wides", "noballs"]
    )
)

print(
    "Legal deliveries:",
    df["legal_delivery"].sum()
)

print(
    "Illegal deliveries:",
    (~df["legal_delivery"]).sum()
)


# ============================================================
# 5. CUMULATIVE RUNS
# ============================================================

print("\n" + "=" * 60)
print("CREATING CUMULATIVE RUN FEATURES")
print("=" * 60)

df["runs_scored"] = (
    df
    .groupby(
        ["match_id", "innings"]
    )["total_runs"]
    .cumsum()
)

print(
    "runs_scored created."
)


# ============================================================
# 6. LEGAL BALLS
# ============================================================

df["legal_balls"] = (
    df
    .groupby(
        ["match_id", "innings"]
    )["legal_delivery"]
    .cumsum()
)

print(
    "legal_balls created."
)


# ============================================================
# 7. WICKETS
# ============================================================

df["wicket"] = (
    df["player_out"].notna()
)

df["wickets_lost"] = (
    df
    .groupby(
        ["match_id", "innings"]
    )["wicket"]
    .cumsum()
)

df["wickets_in_hand"] = (
    10 - df["wickets_lost"]
)

print(
    "wickets_in_hand created."
)


# ============================================================
# 8. TARGET SCORE
# ============================================================

print("\n" + "=" * 60)
print("CREATING TARGET FEATURES")
print("=" * 60)

first_innings_scores = (
    df[
        df["innings"] == 1
    ]
    .groupby("match_id")["total_runs"]
    .sum()
)

target_scores = (
    first_innings_scores + 1
)

df["target"] = (
    df["match_id"]
    .map(target_scores)
)

df["runs_required"] = (
    df["target"]
    - df["runs_scored"]
)

print(
    "target created."
)

print(
    "runs_required created."
)


# ============================================================
# 9. BALLS REMAINING
# ============================================================

# Standard WPL innings:
#
# 20 overs × 6 legal balls = 120 balls

df["balls_remaining"] = (
    120 - df["legal_balls"]
)

# Balls remaining only applies to the
# second innings.

df.loc[
    df["innings"] == 1,
    "balls_remaining"
] = np.nan

print(
    "balls_remaining created."
)


# ============================================================
# 10. CURRENT RUN RATE
# ============================================================

df["current_run_rate"] = (
    df["runs_scored"]
    /
    (df["legal_balls"] / 6)
)

# First innings does not need this
# for our win-probability model.

df.loc[
    df["innings"] == 1,
    "current_run_rate"
] = np.nan

print(
    "current_run_rate created."
)


# ============================================================
# 11. REQUIRED RUN RATE
# ============================================================

df["required_run_rate"] = (
    df["runs_required"]
    /
    (df["balls_remaining"] / 6)
)

df.loc[
    df["innings"] == 1,
    "required_run_rate"
] = np.nan


# Avoid division-by-zero at the end
# of an innings.

df.loc[
    (
        df["innings"] == 2
    )
    &
    (
        df["balls_remaining"] == 0
    ),
    "required_run_rate"
] = 0

print(
    "required_run_rate created."
)


# ============================================================
# 12. MOMENTUM - LAST 12 LEGAL BALLS
# ============================================================

print("\n" + "=" * 60)
print("CREATING MOMENTUM FEATURES")
print("=" * 60)

legal_df = df[
    df["legal_delivery"]
].copy()

legal_df[
    "runs_last_12_balls"
] = (
    legal_df
    .groupby(
        ["match_id", "innings"]
    )["total_runs"]
    .transform(
        lambda x:
        x.rolling(
            window=12,
            min_periods=1
        ).sum()
    )
)

df["runs_last_12_balls"] = pd.NA

df.loc[
    legal_df.index,
    "runs_last_12_balls"
] = (
    legal_df[
        "runs_last_12_balls"
    ]
)

print(
    "runs_last_12_balls created."
)


# ============================================================
# 13. MOMENTUM - LAST 6 LEGAL BALLS
# ============================================================

legal_df[
    "runs_last_6_balls"
] = (
    legal_df
    .groupby(
        ["match_id", "innings"]
    )["total_runs"]
    .transform(
        lambda x:
        x.rolling(
            window=6,
            min_periods=1
        ).sum()
    )
)

df["runs_last_6_balls"] = pd.NA

df.loc[
    legal_df.index,
    "runs_last_6_balls"
] = (
    legal_df[
        "runs_last_6_balls"
    ]
)

print(
    "runs_last_6_balls created."
)


# ============================================================
# 14. MATCH PHASE
# ============================================================

print("\n" + "=" * 60)
print("CREATING MATCH PHASE")
print("=" * 60)


def get_phase(over):

    # Cricsheet over numbers start at 0.

    if over <= 5:
        return "Powerplay"

    elif over <= 14:
        return "Middle"

    else:
        return "Death"


df["phase"] = (
    df["over"].apply(get_phase)
)

print(
    df["phase"].value_counts()
)


# ============================================================
# 15. WIN TARGET
# ============================================================

print("\n" + "=" * 60)
print("CREATING WIN TARGET")
print("=" * 60)

df["win_target"] = pd.NA

# We only model the chasing innings.

chasing_rows = (
    df["innings"] == 2
)

# Ignore matches without a winner.

valid_winner_rows = (
    chasing_rows
    &
    df["winner"].notna()
)

df.loc[
    valid_winner_rows,
    "win_target"
] = (
    (
        df.loc[
            valid_winner_rows,
            "winner"
        ]
        ==
        df.loc[
            valid_winner_rows,
            "batting_team"
        ]
    )
    .astype(int)
)


# ============================================================
# 16. MATCH RULE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("MATCH RULES VALIDATION")
print("=" * 60)


# ------------------------------------------------------------
# Legal balls per innings
# ------------------------------------------------------------

innings_summary = (
    df
    .groupby(
        ["match_id", "innings"]
    )["legal_delivery"]
    .sum()
)

print(
    "\nInnings legal-ball summary:"
)

print(
    innings_summary.describe()
)


# ------------------------------------------------------------
# More than 120 balls
# ------------------------------------------------------------

long_innings = (
    innings_summary[
        innings_summary > 120
    ]
)

print(
    "\nInnings with more than "
    "120 legal balls:"
)

if len(long_innings) > 0:

    print(
        long_innings
        .reset_index(
            name="legal_balls"
        )
    )

else:

    print("None")


# ------------------------------------------------------------
# Exactly 120
# ------------------------------------------------------------

exact_120 = (
    innings_summary == 120
).sum()

print(
    "\nNumber of innings with exactly "
    "120 legal balls:",
    exact_120
)


# ------------------------------------------------------------
# Fewer than 120
# ------------------------------------------------------------

less_than_120 = (
    innings_summary < 120
).sum()

print(
    "\nNumber of innings with fewer "
    "than 120 legal balls:",
    less_than_120
)


# ------------------------------------------------------------
# Unusual matches
# ------------------------------------------------------------

if len(long_innings) > 0:

    unusual_matches = (
        long_innings
        .reset_index()["match_id"]
        .astype(str)
        .unique()
        .tolist()
    )

else:

    unusual_matches = []


print(
    "\nMatches with innings longer "
    "than 120 legal balls:"
)

print(
    unusual_matches
)


# ------------------------------------------------------------
# Negative balls remaining
# ------------------------------------------------------------

negative_balls = df[
    df["balls_remaining"] < 0
]

print(
    "\nRows with negative "
    "balls remaining:",
    len(negative_balls)
)


# ------------------------------------------------------------
# Zero balls remaining
# ------------------------------------------------------------

zero_balls = df[
    df["balls_remaining"] == 0
]

print(
    "Rows with zero balls remaining:",
    len(zero_balls)
)


# ------------------------------------------------------------
# Negative runs required
# ------------------------------------------------------------

negative_runs = df[
    (df["innings"] == 2)
    &
    (df["runs_required"] < 0)
]

print(
    "Rows with negative "
    "runs required:",
    len(negative_runs)
)


# ------------------------------------------------------------
# Zero runs required
# ------------------------------------------------------------

zero_runs = df[
    (df["innings"] == 2)
    &
    (df["runs_required"] == 0)
]

print(
    "Rows with zero "
    "runs required:",
    len(zero_runs)
)


# ============================================================
# 17. TARGET CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("TARGET CONSISTENCY CHECK")
print("=" * 60)

recalculated_target = (
    df["runs_scored"]
    +
    df["runs_required"]
)

target_check = (
    recalculated_target
    ==
    df["target"]
)

print(
    target_check.value_counts()
)

target_mismatches = df[
    ~target_check
]

print(
    "Target mismatches:",
    len(target_mismatches)
)


# ============================================================
# 18. MISSING WINNERS
# ============================================================

missing_winner_matches = (
    df[
        df["winner"].isna()
    ]["match_id"]
    .unique()
)

print(
    "\nMatches without winner:",
    len(missing_winner_matches)
)

print(
    missing_winner_matches
)


# ============================================================
# 19. VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(
    "Total matches:",
    df["match_id"].nunique()
)

print(
    "Innings >120:",
    len(long_innings)
)

print(
    "Innings <120:",
    less_than_120
)

print(
    "Negative balls remaining:",
    len(negative_balls)
)

print(
    "Negative runs required:",
    len(negative_runs)
)

print(
    "Zero runs required:",
    len(zero_runs)
)

print(
    "Target mismatches:",
    len(target_mismatches)
)

print(
    "Matches without winner:",
    len(missing_winner_matches)
)


# ============================================================
# 20. CREATE MODELING DATASET
# ============================================================

print("\n" + "=" * 60)
print("CREATING MODELING DATASET")
print("=" * 60)

# We only want:
#
# innings == 2
# valid winner
# runs_required > 0
#
# The last condition removes states where the
# chasing team has already passed the target.

model_df = df[
    (df["innings"] == 2)
    &
    (df["win_target"].notna())
    &
    (df["runs_required"] > 0)
].copy()

print(
    "Modeling dataset shape:",
    model_df.shape
)

print(
    "Unique matches:",
    model_df["match_id"].nunique()
)

print(
    "\nTarget distribution:"
)

print(
    model_df[
        "win_target"
    ].value_counts()
)


# ============================================================
# 21. LEGAL DELIVERIES ONLY
# ============================================================

trainable_df = model_df[
    model_df["legal_delivery"]
].copy()

print(
    "\nAfter legal-delivery filtering:",
    trainable_df.shape
)


# ============================================================
# 22. MATCH-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("MATCH-LEVEL TRAIN / TEST SPLIT")
print("=" * 60)

match_ids = (
    trainable_df[
        "match_id"
    ]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

train_matches, test_matches = (
    train_test_split(
        match_ids,
        test_size=0.20,
        random_state=42
    )
)


train_df = trainable_df[
    trainable_df[
        "match_id"
    ]
    .astype(str)
    .isin(train_matches)
].copy()


test_df = trainable_df[
    trainable_df[
        "match_id"
    ]
    .astype(str)
    .isin(test_matches)
].copy()


print(
    "Total unique matches:",
    trainable_df[
        "match_id"
    ].nunique()
)

print(
    "Training matches:",
    len(train_matches)
)

print(
    "Testing matches:",
    len(test_matches)
)


# ------------------------------------------------------------
# Check match leakage
# ------------------------------------------------------------

common_matches = (
    set(train_matches)
    &
    set(test_matches)
)

print(
    "Common matches:",
    len(common_matches)
)


# ============================================================
# 23. FEATURES
# ============================================================

numeric_features = [

    "runs_scored",

    "runs_required",

    "balls_remaining",

    "wickets_in_hand",

    "current_run_rate",

    "required_run_rate",

    "runs_last_12_balls",

    "runs_last_6_balls"

]


categorical_features = [

    "phase"

]


print(
    "\nNumeric features:"
)

for feature in numeric_features:

    print(
        "-",
        feature
    )


print(
    "\nCategorical features:"
)

for feature in categorical_features:

    print(
        "-",
        feature
    )


# ============================================================
# 24. X AND Y
# ============================================================

X_train = train_df[
    numeric_features
    +
    categorical_features
].copy()

X_test = test_df[
    numeric_features
    +
    categorical_features
].copy()


y_train = train_df[
    "win_target"
].astype(int)


y_test = test_df[
    "win_target"
].astype(int)


# ============================================================
# 25. SAFETY FOR REQUIRED RUN RATE
# ============================================================

train_df.loc[
    train_df["balls_remaining"] == 0,
    "required_run_rate"
] = 0

test_df.loc[
    test_df["balls_remaining"] == 0,
    "required_run_rate"
] = 0


# Recreate X after correction.

X_train = train_df[
    numeric_features
    +
    categorical_features
].copy()

X_test = test_df[
    numeric_features
    +
    categorical_features
].copy()


# Replace infinite values.

X_train = X_train.replace(
    [np.inf, -np.inf],
    0
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    0
)


# ============================================================
# 26. FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL FEATURE VALIDATION")
print("=" * 60)

print(
    "\nTraining missing values:"
)

print(
    X_train.isna().sum()
)


print(
    "\nTesting missing values:"
)

print(
    X_test.isna().sum()
)


# ============================================================
# 27. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            "passthrough",

            numeric_features
        ),

        (
            "phase",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ]

)


# ============================================================
# 28. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=1000
            )
        )

    ]

)


model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression training complete."
)


# ============================================================
# 29. ORIGINAL MODEL PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 30. ORIGINAL MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION EVALUATION")
print("=" * 60)


accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

logloss = log_loss(
    y_test,
    y_prob
)

brier = brier_score_loss(
    y_test,
    y_prob
)


print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"Log Loss: {logloss:.4f}"
)

print(
    f"Brier Score: {brier:.4f}"
)


# ============================================================
# 31. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    cm
)


# ============================================================
# 32. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 33. ORIGINAL CALIBRATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION ANALYSIS")
print("=" * 60)

prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="uniform"
)


calibration_table = pd.DataFrame({

    "predicted_probability":
        prob_pred,

    "actual_win_rate":
        prob_true

})


print(
    calibration_table
)


# ============================================================
# 34. LOGISTIC REGRESSION CALIBRATION CURVE
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel(
    "Predicted Win Probability"
)

plt.ylabel(
    "Actual Win Rate"
)

plt.title(
    "WPL Win Probability Calibration"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ============================================================
# 35. LOGISTIC REGRESSION FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
print("=" * 60)

classifier = (
    model.named_steps[
        "classifier"
    ]
)

fitted_preprocessor = (
    model.named_steps[
        "preprocessor"
    ]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

coefficients = (
    classifier.coef_[0]
)

feature_importance = (
    pd.DataFrame({

        "feature":
            feature_names,

        "coefficient":
            coefficients,

        "absolute_coefficient":
            np.abs(coefficients)

    })
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)

print(
    feature_importance
)


# ============================================================
# 36. XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

try:

    from xgboost import XGBClassifier


    xgb_model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",

                XGBClassifier(

                    n_estimators=300,

                    max_depth=5,

                    learning_rate=0.05,

                    subsample=0.8,

                    colsample_bytree=0.8,

                    objective="binary:logistic",

                    eval_metric="logloss",

                    random_state=42

                )

            )

        ]

    )


    xgb_model.fit(
        X_train,
        y_train
    )


    xgb_pred = (
        xgb_model.predict(
            X_test
        )
    )


    xgb_prob = (
        xgb_model
        .predict_proba(
            X_test
        )[:, 1]
    )


    xgb_accuracy = (
        accuracy_score(
            y_test,
            xgb_pred
        )
    )


    xgb_roc_auc = (
        roc_auc_score(
            y_test,
            xgb_prob
        )
    )


    xgb_logloss = (
        log_loss(
            y_test,
            xgb_prob
        )
    )


    xgb_brier = (
        brier_score_loss(
            y_test,
            xgb_prob
        )
    )


    print(
        "\nXGBoost Results:"
    )

    print(
        f"Accuracy: {xgb_accuracy:.4f}"
    )

    print(
        f"ROC-AUC: {xgb_roc_auc:.4f}"
    )

    print(
        f"Log Loss: {xgb_logloss:.4f}"
    )

    print(
        f"Brier Score: {xgb_brier:.4f}"
    )


    print(
        "\nXGBoost Classification Report:"
    )

    print(
        classification_report(
            y_test,
            xgb_pred
        )
    )


except ImportError:

    print(
        "\nXGBoost is not installed."
    )

    print(
        "Install using:"
    )

    print(
        "python -m pip install xgboost"
    )


# ============================================================
# 37. ORIGINAL MODEL VS XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


try:

    model_comparison = pd.DataFrame({

        "Model": [

            "Logistic Regression",

            "XGBoost"

        ],

        "Accuracy": [

            accuracy,

            xgb_accuracy

        ],

        "ROC-AUC": [

            roc_auc,

            xgb_roc_auc

        ],

        "Log Loss": [

            logloss,

            xgb_logloss

        ],

        "Brier Score": [

            brier,

            xgb_brier

        ]

    })


    print(
        model_comparison.to_string(
            index=False
        )
    )


except NameError:

    print(
        "XGBoost comparison unavailable."
    )


# ============================================================
# 38. PROBABILITY CALIBRATION
# ============================================================

print("\n" + "=" * 60)
print("PROBABILITY CALIBRATION EXPERIMENT")
print("=" * 60)


# We use the complete Logistic Regression pipeline
# inside CalibratedClassifierCV.
#
# sigmoid = Platt scaling.
#
# cv=5 means calibration is learned using
# cross-validation on the training set.
#
# The test set remains untouched.

calibrated_model = (
    CalibratedClassifierCV(

        estimator=Pipeline(

            steps=[

                (
                    "preprocessor",

                    preprocessor
                ),

                (
                    "classifier",

                    LogisticRegression(
                        max_iter=1000
                    )
                )

            ]

        ),

        method="sigmoid",

        cv=5

    )
)


# ============================================================
# 39. TRAIN CALIBRATED MODEL
# ============================================================

calibrated_model.fit(
    X_train,
    y_train
)

print(
    "Calibrated Logistic Regression "
    "training complete."
)


# ============================================================
# 40. CALIBRATED PREDICTIONS
# ============================================================

calibrated_pred = (
    calibrated_model.predict(
        X_test
    )
)

calibrated_prob = (
    calibrated_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 41. CALIBRATED MODEL METRICS
# ============================================================

calibrated_accuracy = (
    accuracy_score(
        y_test,
        calibrated_pred
    )
)

calibrated_roc_auc = (
    roc_auc_score(
        y_test,
        calibrated_prob
    )
)

calibrated_logloss = (
    log_loss(
        y_test,
        calibrated_prob
    )
)

calibrated_brier = (
    brier_score_loss(
        y_test,
        calibrated_prob
    )
)


print(
    "\nCalibrated Logistic Regression:"
)

print(
    f"Accuracy: {calibrated_accuracy:.4f}"
)

print(
    f"ROC-AUC: {calibrated_roc_auc:.4f}"
)

print(
    f"Log Loss: {calibrated_logloss:.4f}"
)

print(
    f"Brier Score: {calibrated_brier:.4f}"
)


# ============================================================
# 42. CALIBRATION COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION COMPARISON")
print("=" * 60)


calibration_comparison = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "ROC-AUC",

        "Log Loss",

        "Brier Score"

    ],

    "Original Logistic Regression": [

        accuracy,

        roc_auc,

        logloss,

        brier

    ],

    "Calibrated Logistic Regression": [

        calibrated_accuracy,

        calibrated_roc_auc,

        calibrated_logloss,

        calibrated_brier

    ]

})


print(
    calibration_comparison.to_string(
        index=False
    )
)


# ============================================================
# 43. CALIBRATED MODEL CALIBRATION CURVE
# ============================================================

cal_prob_true, cal_prob_pred = (
    calibration_curve(

        y_test,

        calibrated_prob,

        n_bins=10,

        strategy="uniform"

    )
)


# ============================================================
# 44. COMBINED CALIBRATION GRAPH
# ============================================================

print("\n" + "=" * 60)
print("CREATING CALIBRATION COMPARISON GRAPH")
print("=" * 60)


plt.figure(
    figsize=(8, 8)
)


# Original model

plt.plot(

    prob_pred,

    prob_true,

    marker="o",

    label="Original Logistic Regression"

)


# Calibrated model

plt.plot(

    cal_prob_pred,

    cal_prob_true,

    marker="s",

    label="Calibrated Logistic Regression"

)


# Perfect calibration

plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Perfect Calibration"

)


plt.xlabel(
    "Predicted Win Probability"
)

plt.ylabel(
    "Actual Win Rate"
)

plt.title(
    "WPL Win Probability Calibration Comparison"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ============================================================
# 45. CALIBRATION DECISION
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION DECISION")
print("=" * 60)


if calibrated_brier < brier:

    print(
        "Calibrated model has a LOWER Brier Score."
    )

    brier_decision = (
        "Calibrated model"
    )

else:

    print(
        "Original model has a LOWER Brier Score."
    )

    brier_decision = (
        "Original model"
    )


if calibrated_logloss < logloss:

    print(
        "Calibrated model has a LOWER Log Loss."
    )

    logloss_decision = (
        "Calibrated model"
    )

else:

    print(
        "Original model has a LOWER Log Loss."
    )

    logloss_decision = (
        "Original model"
    )


# ============================================================
# 46. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)


print(
    "\nDataset:"
)

print(
    "Total raw deliveries:",
    len(df)
)

print(
    "Total matches:",
    df["match_id"].nunique()
)

print(
    "Modeling matches:",
    trainable_df[
        "match_id"
    ].nunique()
)

print(
    "Training matches:",
    len(train_matches)
)

print(
    "Testing matches:",
    len(test_matches)
)


print(
    "\nOriginal Logistic Regression:"
)

print(
    f"Accuracy    = {accuracy:.4f}"
)

print(
    f"ROC-AUC     = {roc_auc:.4f}"
)

print(
    f"Log Loss    = {logloss:.4f}"
)

print(
    f"Brier Score = {brier:.4f}"
)


print(
    "\nCalibrated Logistic Regression:"
)

print(
    f"Accuracy    = {calibrated_accuracy:.4f}"
)

print(
    f"ROC-AUC     = {calibrated_roc_auc:.4f}"
)

print(
    f"Log Loss    = {calibrated_logloss:.4f}"
)

print(
    f"Brier Score = {calibrated_brier:.4f}"
)


print(
    "\nCalibration comparison:"
)

print(
    "Brier Score:",
    brier_decision
)

print(
    "Log Loss:",
    logloss_decision
)


print("\n" + "=" * 60)
print(
    "MODEL TRAINING AND EVALUATION COMPLETED"
)
print("=" * 60)

# ============================================================
# 47. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING FINAL MODEL")
print("=" * 60)

import joblib
import json
import os


# ------------------------------------------------------------
# Model directory
# ------------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


# ------------------------------------------------------------
# Save the ORIGINAL Logistic Regression pipeline
# ------------------------------------------------------------

model_path = (
    "models/"
    "wpl_win_probability_model.joblib"
)

joblib.dump(
    model,
    model_path
)

print(
    "Model saved to:",
    model_path
)


# ------------------------------------------------------------
# Save model metadata
# ------------------------------------------------------------

metadata = {

    "model_type":
        "Logistic Regression",

    "dataset":
        "Cricsheet WPL",

    "total_matches":
        int(df["match_id"].nunique()),

    "modeling_matches":
        int(
            trainable_df[
                "match_id"
            ].nunique()
        ),

    "training_matches":
        int(
            len(train_matches)
        ),

    "testing_matches":
        int(
            len(test_matches)
        ),

    "accuracy":
        float(accuracy),

    "roc_auc":
        float(roc_auc),

    "log_loss":
        float(logloss),

    "brier_score":
        float(brier),

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "target":
        "win_target",

    "state_type":
        "post_delivery",

    "probability_calibration":
        "Original Logistic Regression",

    "random_state":
        42

}


metadata_path = (
    "models/"
    "model_metadata.json"
)


with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    "Metadata saved to:",
    metadata_path
)


print("\n" + "=" * 60)
print("MODEL SAVING COMPLETED")
print("=" * 60)