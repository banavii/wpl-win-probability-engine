# ============================================================
# WPL WIN PROBABILITY PREDICTION ENGINE
# ============================================================

import joblib
import pandas as pd


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = (
    "models/"
    "wpl_win_probability_model.joblib"
)

model = joblib.load(
    MODEL_PATH
)

print(
    "\nWPL Win Probability Engine"
)

print(
    "-" * 50
)


# ============================================================
# 2. GET MATCH INPUT
# ============================================================

print(
    "\nEnter the current match state."
)

print(
    "All values should represent the state "
    "AFTER the latest legal delivery."
)


# ------------------------------------------------------------
# Runs scored
# ------------------------------------------------------------

runs_scored = float(
    input(
        "\nRuns scored: "
    )
)


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

target = float(
    input(
        "Target: "
    )
)


# ------------------------------------------------------------
# Legal balls completed
# ------------------------------------------------------------

legal_balls = int(
    input(
        "Legal balls completed: "
    )
)


# ------------------------------------------------------------
# Wickets lost
# ------------------------------------------------------------

wickets_lost = int(
    input(
        "Wickets lost: "
    )
)


# ------------------------------------------------------------
# Recent runs
# ------------------------------------------------------------

runs_last_12_balls = float(
    input(
        "Runs scored in last 12 legal balls: "
    )
)

runs_last_6_balls = float(
    input(
        "Runs scored in last 6 legal balls: "
    )
)


# ------------------------------------------------------------
# Current over
# ------------------------------------------------------------

over_number = int(
    input(
        "Current over number (0-19): "
    )
)


# ============================================================
# 3. DERIVE FEATURES
# ============================================================

runs_required = (
    target - runs_scored
)


balls_remaining = (
    120 - legal_balls
)


wickets_in_hand = (
    10 - wickets_lost
)


# ------------------------------------------------------------
# Current Run Rate
# ------------------------------------------------------------

if legal_balls > 0:

    current_run_rate = (
        runs_scored
        /
        (legal_balls / 6)
    )

else:

    current_run_rate = 0


# ------------------------------------------------------------
# Required Run Rate
# ------------------------------------------------------------

if balls_remaining > 0:

    required_run_rate = (
        runs_required
        /
        (balls_remaining / 6)
    )

else:

    required_run_rate = 0


# ============================================================
# 4. DETERMINE MATCH PHASE
# ============================================================

if over_number <= 5:

    phase = "Powerplay"

elif over_number <= 14:

    phase = "Middle"

else:

    phase = "Death"


# ============================================================
# 5. DISPLAY CALCULATED STATE
# ============================================================

print(
    "\n" + "=" * 50
)

print(
    "CALCULATED MATCH STATE"
)

print(
    "=" * 50
)

print(
    f"Runs scored:          {runs_scored:.0f}"
)

print(
    f"Runs required:        {runs_required:.0f}"
)

print(
    f"Balls remaining:      {balls_remaining}"
)

print(
    f"Wickets in hand:      {wickets_in_hand}"
)

print(
    f"Current run rate:     {current_run_rate:.2f}"
)

print(
    f"Required run rate:    {required_run_rate:.2f}"
)

print(
    f"Runs last 12 balls:   {runs_last_12_balls:.0f}"
)

print(
    f"Runs last 6 balls:    {runs_last_6_balls:.0f}"
)

print(
    f"Phase:                {phase}"
)


# ============================================================
# 6. CREATE MODEL INPUT
# ============================================================

input_data = pd.DataFrame({

    "runs_scored": [
        runs_scored
    ],

    "runs_required": [
        runs_required
    ],

    "balls_remaining": [
        balls_remaining
    ],

    "wickets_in_hand": [
        wickets_in_hand
    ],

    "current_run_rate": [
        current_run_rate
    ],

    "required_run_rate": [
        required_run_rate
    ],

    "runs_last_12_balls": [
        runs_last_12_balls
    ],

    "runs_last_6_balls": [
        runs_last_6_balls
    ],

    "phase": [
        phase
    ]

})


# ============================================================
# 7. PREDICT WIN PROBABILITY
# ============================================================

win_probability = (
    model
    .predict_proba(
        input_data
    )[0][1]
)


# ============================================================
# 8. OPPOSITION PROBABILITY
# ============================================================

loss_probability = (
    1 - win_probability
)


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

print(
    "\n" + "=" * 50
)

print(
    "WPL WIN PROBABILITY"
)

print(
    "=" * 50
)

print(
    f"\nBatting team: "
    f"{win_probability * 100:.2f}%"
)

print(
    f"Opposition:   "
    f"{loss_probability * 100:.2f}%"
)


# ============================================================
# 10. SIMPLE INTERPRETATION
# ============================================================

if win_probability >= 0.80:

    interpretation = (
        "Strong winning position"
    )

elif win_probability >= 0.60:

    interpretation = (
        "Favorable position"
    )

elif win_probability >= 0.40:

    interpretation = (
        "Competitive / uncertain position"
    )

elif win_probability >= 0.20:

    interpretation = (
        "Challenging position"
    )

else:

    interpretation = (
        "Difficult winning position"
    )


print(
    "\nMatch state:",
    interpretation
)

print(
    "\n" + "=" * 50
)