import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predictor import WPLPredictor
from src.replay import replay_second_innings


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="WPL Win Probability Engine",
    page_icon="🏏",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_predictor():
    return WPLPredictor()


@st.cache_data
def load_data():
    data_path = ROOT_DIR / "data" / "processed" / "deliveries.parquet"
    return pd.read_parquet(data_path)


predictor = load_predictor()
df = load_data()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🏏 WPL Cricket Win Probability Engine")

st.markdown(
    """
    **Cricket Win Probability & Match Analytics Engine**

    Predict the chasing team's win probability from the current
    match state and replay historical WPL matches delivery by delivery.
    """
)

st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

historical_tab, live_tab = st.tabs(
    [
        "📈 Historical Match Replay",
        "🔴 Live Match Prediction"
    ]
)


# =========================================================
# HISTORICAL MATCH REPLAY
# =========================================================

with historical_tab:

    st.header("📈 Historical Match Replay")

    st.markdown(
        """
        Select a completed WPL match to replay the second innings.
        The model generates a win probability after every legal delivery.
        """
    )

    # -----------------------------------------------------
    # MATCH LIST
    # -----------------------------------------------------

    match_info = (
        df[
            [
                "match_id",
                "date",
                "team1",
                "team2",
                "winner"
            ]
        ]
        .drop_duplicates("match_id")
    )

    # Only matches with a winner
    match_info = match_info[
        match_info["winner"].notna()
    ].copy()

    match_info["label"] = (
        match_info["team1"]
        + " vs "
        + match_info["team2"]
        + " | "
        + match_info["date"].astype(str)
        + " | Match ID: "
        + match_info["match_id"].astype(str)
    )

    match_options = dict(
        zip(
            match_info["label"],
            match_info["match_id"]
        )
    )

    selected_match_label = st.selectbox(
        "Select a WPL match",
        options=list(match_options.keys())
    )

    selected_match_id = match_options[selected_match_label]

    # -----------------------------------------------------
    # MATCH INFORMATION
    # -----------------------------------------------------

    selected_match = match_info[
        match_info["match_id"] == selected_match_id
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Team 1",
            selected_match["team1"]
        )

    with col2:
        st.metric(
            "Team 2",
            selected_match["team2"]
        )

    with col3:
        st.metric(
            "Winner",
            selected_match["winner"]
        )

    st.write("")

    # -----------------------------------------------------
    # REPLAY BUTTON
    # -----------------------------------------------------

    replay_button = st.button(
        "▶️ Replay Second Innings",
        type="primary",
        use_container_width=True
    )

    if replay_button:

        with st.spinner("Replaying match..."):

            try:

                replay_df = replay_second_innings(
                    selected_match_id
                )

                st.session_state["replay_df"] = replay_df
                st.session_state["replay_match_id"] = selected_match_id

            except Exception as e:

                st.error(
                    f"Unable to replay this match: {e}"
                )

    # -----------------------------------------------------
    # DISPLAY REPLAY
    # -----------------------------------------------------

    if (
        "replay_df" in st.session_state
        and st.session_state.get("replay_match_id")
        == selected_match_id
    ):

        replay_df = st.session_state["replay_df"]

        if replay_df.empty:

            st.warning(
                "No replay data is available for this match."
            )

        else:

            st.success(
                f"Replay generated successfully: "
                f"{len(replay_df)} delivery states"
            )

            # -------------------------------------------------
            # WIN PROBABILITY CHART
            # -------------------------------------------------

            st.subheader(
                "📊 Win Probability Throughout the Chase"
            )

            chart_df = replay_df[
                [
                    "legal_balls",
                    "win_probability"
                ]
            ].copy()

            chart_df["win_probability"] *= 100

            chart_df = chart_df.rename(
                columns={
                    "legal_balls": "Legal Balls",
                    "win_probability": "Batting Team Win Probability (%)"
                }
            )

            st.line_chart(
                chart_df.set_index("Legal Balls"),
                use_container_width=True
            )

            # -------------------------------------------------
            # LATEST STATE
            # -------------------------------------------------

            latest = replay_df.iloc[-1]

            st.subheader("🎯 Latest Match State")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Runs",
                    int(latest["runs_scored"])
                )

            with col2:
                st.metric(
                    "Runs Required",
                    max(0, int(latest["runs_required"]))
                )

            with col3:
                st.metric(
                    "Wickets in Hand",
                    int(latest["wickets_in_hand"])
                )

            with col4:
                st.metric(
                    "Balls Remaining",
                    int(latest["balls_remaining"])
                )

            # -------------------------------------------------
            # CURRENT PROBABILITY
            # -------------------------------------------------

            probability = latest["win_probability"] * 100

            st.subheader("🏆 Current Win Probability")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Chasing Team",
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(max(probability / 100, 0), 1)
                )

            with col2:

                opposition_probability = 100 - probability

                st.metric(
                    "Opposition",
                    f"{opposition_probability:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            opposition_probability / 100,
                            0
                        ),
                        1
                    )
                )

            # -------------------------------------------------
            # ANALYTICS
            # -------------------------------------------------

            st.subheader("📊 Match Analytics")

            col1, col2, col3 = st.columns(3)

            with col1:

                max_probability = (
                    replay_df["win_probability"].max()
                    * 100
                )

                st.metric(
                    "Highest Chasing Probability",
                    f"{max_probability:.2f}%"
                )

            with col2:

                min_probability = (
                    replay_df["win_probability"].min()
                    * 100
                )

                st.metric(
                    "Lowest Chasing Probability",
                    f"{min_probability:.2f}%"
                )

            with col3:

                probability_change = (
                    replay_df["win_probability"].iloc[-1]
                    - replay_df["win_probability"].iloc[0]
                ) * 100

                st.metric(
                    "Probability Change",
                    f"{probability_change:+.2f}%"
                )

            # -------------------------------------------------
            # DELIVERY TABLE
            # -------------------------------------------------

            with st.expander(
                "📋 View Delivery-by-Delivery Data"
            ):

                display_replay = replay_df.copy()

                if "win_probability" in display_replay.columns:

                    display_replay["win_probability"] = (
                        display_replay["win_probability"] * 100
                    ).round(2)

                st.dataframe(
                    display_replay,
                    use_container_width=True,
                    hide_index=True
                )


# =========================================================
# LIVE MATCH PREDICTION
# =========================================================

with live_tab:

    st.header("🔴 Live Match Prediction")

    st.markdown(
        """
        Enter the current state of a T20 chase to estimate the
        batting team's win probability.
        """
    )

    # -----------------------------------------------------
    # TEAM INFORMATION
    # -----------------------------------------------------

    st.subheader("🏏 Match Information")

    col1, col2 = st.columns(2)

    with col1:

        batting_team = st.text_input(
            "Batting Team",
            value="Chasing Team"
        )

    with col2:

        opposition = st.text_input(
            "Opposition",
            value="Opposition"
        )

    # -----------------------------------------------------
    # SCORE INFORMATION
    # -----------------------------------------------------

    st.subheader("📌 Current Match State")

    col1, col2, col3 = st.columns(3)

    with col1:

        runs_scored = st.number_input(
            "Runs Scored",
            min_value=0,
            max_value=500,
            value=85,
            step=1
        )

    with col2:

        wickets_lost = st.number_input(
            "Wickets Lost",
            min_value=0,
            max_value=10,
            value=3,
            step=1
        )

    with col3:

        target = st.number_input(
            "Target",
            min_value=1,
            max_value=500,
            value=161,
            step=1
        )

    # -----------------------------------------------------
    # BALL INFORMATION
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        legal_balls = st.number_input(
            "Legal Balls Completed",
            min_value=0,
            max_value=120,
            value=72,
            step=1
        )

    with col2:

        current_over = st.number_input(
            "Current Over",
            min_value=0,
            max_value=19,
            value=11,
            step=1
        )

    with col3:

        st.metric(
            "Balls Remaining",
            max(0, 120 - legal_balls)
        )

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    st.subheader("⚡ Recent Batting Momentum")

    col1, col2 = st.columns(2)

    with col1:

        runs_last_12_balls = st.number_input(
            "Runs in Last 12 Legal Balls",
            min_value=0,
            max_value=100,
            value=18,
            step=1
        )

    with col2:

        runs_last_6_balls = st.number_input(
            "Runs in Last 6 Legal Balls",
            min_value=0,
            max_value=60,
            value=10,
            step=1
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    st.subheader("✅ Input Validation")

    validation_errors = []

    if runs_scored > target:
        validation_errors.append(
            "Runs scored cannot be greater than the target."
        )

    if wickets_lost > 10:
        validation_errors.append(
            "Wickets lost cannot be greater than 10."
        )

    if legal_balls > 120:
        validation_errors.append(
            "Legal balls cannot exceed 120 for a standard T20 innings."
        )

    if runs_last_6_balls > runs_last_12_balls:
        validation_errors.append(
            "Runs in the last 6 balls cannot exceed "
            "runs in the last 12 balls."
        )

    if validation_errors:

        for error in validation_errors:
            st.error(error)

        valid_input = False

    else:

        st.success(
            "Match state is valid."
        )

        valid_input = True

    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    st.write("")

    predict_button = st.button(
        "🔮 Calculate Win Probability",
        type="primary",
        use_container_width=True
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if predict_button and valid_input:

        result = predictor.predict(
            runs_scored=runs_scored,
            target=target,
            legal_balls=legal_balls,
            wickets_lost=wickets_lost,
            runs_last_12_balls=runs_last_12_balls,
            runs_last_6_balls=runs_last_6_balls,
            current_over=current_over
        )

        st.session_state["live_result"] = result
        st.session_state["live_batting_team"] = batting_team
        st.session_state["live_opposition"] = opposition

    # -----------------------------------------------------
    # DISPLAY PREDICTION
    # -----------------------------------------------------

    if "live_result" in st.session_state:

        result = st.session_state["live_result"]

        batting_team = st.session_state.get(
            "live_batting_team",
            batting_team
        )

        opposition = st.session_state.get(
            "live_opposition",
            opposition
        )

        probability = (
            result["batting_team_probability"] * 100
        )

        opposition_probability = (
            result["opposition_probability"] * 100
        )

        st.divider()

        st.subheader(
            "🏆 Win Probability"
        )

        # -------------------------------------------------
        # PROBABILITY DISPLAY
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                batting_team,
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability / 100,
                        0
                    ),
                    1
                )
            )

        with col2:

            st.metric(
                opposition,
                f"{opposition_probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        opposition_probability / 100,
                        0
                    ),
                    1
                )
            )

        # -------------------------------------------------
        # MATCH STATE
        # -------------------------------------------------

        st.subheader(
            "📊 Match State"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Runs Required",
                int(result["runs_required"])
            )

        with col2:

            st.metric(
                "Balls Remaining",
                int(result["balls_remaining"])
            )

        with col3:

            st.metric(
                "Wickets in Hand",
                int(result["wickets_in_hand"])
            )

        with col4:

            st.metric(
                "Phase",
                result["phase"]
            )

        # -------------------------------------------------
        # ANALYTICS
        # -------------------------------------------------

        st.subheader(
            "📈 Match Analytics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Current Run Rate",
                f"{result['current_run_rate']:.2f}"
            )

        with col2:

            st.metric(
                "Required Run Rate",
                f"{result['required_run_rate']:.2f}"
            )

        with col3:

            st.metric(
                "Runs Required",
                int(result["runs_required"])
            )

        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        st.subheader(
            "🧠 Prediction Interpretation"
        )

        if probability >= 70:

            st.success(
                f"The model currently estimates a "
                f"{probability:.2f}% win probability "
                f"for {batting_team}."
            )

        elif probability >= 50:

            st.info(
                f"The match is relatively balanced, with "
                f"{batting_team} at {probability:.2f}%."
            )

        else:

            st.warning(
                f"The model currently estimates a "
                f"{probability:.2f}% win probability "
                f"for {batting_team}."
            )

        # -------------------------------------------------
        # FEATURE INPUTS
        # -------------------------------------------------

        with st.expander(
            "🔍 View Model Features"
        ):

            feature_display = pd.DataFrame(
                {
                    "Feature": [
                        "Runs Scored",
                        "Runs Required",
                        "Balls Remaining",
                        "Wickets in Hand",
                        "Current Run Rate",
                        "Required Run Rate",
                        "Runs Last 12 Balls",
                        "Runs Last 6 Balls",
                        "Phase"
                    ],
                    "Value": [
                        result["runs_scored"],
                        result["runs_required"],
                        result["balls_remaining"],
                        result["wickets_in_hand"],
                        round(
                            result["current_run_rate"],
                            3
                        ),
                        round(
                            result["required_run_rate"],
                            3
                        ),
                        result["runs_last_12_balls"],
                        result["runs_last_6_balls"],
                        result["phase"]
                    ]
                }
            )

            st.dataframe(
                feature_display,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # MODEL EXPLANATION
        # -------------------------------------------------

        st.subheader(
            "🧠 Why Did the Model Make This Prediction?"
        )

        st.caption(
            "These are the model's feature contributions for "
            "the current match state. Positive values push the "
            "prediction toward the batting team; negative values "
            "push it away. Contributions are in model "
            "linear-predictor space, not percentage points."
        )

        # Get contribution data
        contribution_df = (
            predictor.get_prediction_contributions(
                result
            )
        )

        # Positive contributions
        positive_contributions = contribution_df[
            contribution_df["contribution"] > 0
        ].copy()

        # Negative contributions
        negative_contributions = contribution_df[
            contribution_df["contribution"] < 0
        ].copy()

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # POSITIVE FACTORS
        # -------------------------------------------------

        with col1:

            st.markdown(
                "### 📈 Positive Factors"
            )

            positive_display = (
                positive_contributions[
                    [
                        "feature",
                        "value",
                        "contribution"
                    ]
                ]
                .head(5)
                .copy()
            )

            positive_display["contribution"] = (
                positive_display["contribution"]
                .round(3)
            )

            positive_display["value"] = (
                positive_display["value"]
                .round(3)
            )

            positive_display = positive_display.rename(
                columns={
                    "feature": "Feature",
                    "value": "Current Value",
                    "contribution": "Contribution"
                }
            )

            st.dataframe(
                positive_display,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # NEGATIVE FACTORS
        # -------------------------------------------------

        with col2:

            st.markdown(
                "### 📉 Negative Factors"
            )

            negative_display = (
                negative_contributions[
                    [
                        "feature",
                        "value",
                        "contribution"
                    ]
                ]
                .head(5)
                .copy()
            )

            negative_display["contribution"] = (
                negative_display["contribution"]
                .round(3)
            )

            negative_display["value"] = (
                negative_display["value"]
                .round(3)
            )

            negative_display = negative_display.rename(
                columns={
                    "feature": "Feature",
                    "value": "Current Value",
                    "contribution": "Contribution"
                }
            )

            st.dataframe(
                negative_display,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # MODEL COEFFICIENTS
        # -------------------------------------------------

        with st.expander(
            "📚 View Global Model Feature Effects"
        ):

            coefficients = (
                predictor.explain_prediction()
            )

            coefficients["coefficient"] = (
                coefficients["coefficient"]
                .round(4)
            )

            coefficients = coefficients.rename(
                columns={
                    "feature": "Feature",
                    "coefficient": "Coefficient",
                    "effect": "Effect"
                }
            )

            st.dataframe(
                coefficients[
                    [
                        "Feature",
                        "Coefficient",
                        "Effect"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Global coefficients describe the Logistic Regression "
                "model's learned direction of association. They are "
                "not the same as the contribution of a feature for "
                "one specific match state."
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "WPL Cricket Win Probability & Match Analytics Engine | "
    "Model-derived probabilities for analytical purposes"
)