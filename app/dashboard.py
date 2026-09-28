import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predictor import WPLPredictor
from src.replay import replay_second_innings


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="WPL Win Probability Engine",
    page_icon="🏏",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_predictor():
    return WPLPredictor()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data_path = (
        ROOT_DIR
        / "data"
        / "processed"
        / "deliveries.parquet"
    )

    return pd.read_parquet(data_path)


predictor = load_predictor()
df = load_data()


# =========================================================
# HEADER
# =========================================================

st.title("🏏 WPL Cricket Win Probability Engine")

st.markdown(
    """
**Cricket Win Probability & Match Analytics Engine**

Predict the chasing team's win probability from the current
match state and replay historical WPL matches delivery by delivery.
"""
)

st.divider()


# =========================================================
# TABS
# =========================================================

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

    selected_match_id = match_options[
        selected_match_label
    ]

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
        width="stretch"
    )

    if replay_button:

        with st.spinner("Replaying match..."):

            try:

                replay_df = replay_second_innings(
                    selected_match_id
                )

                st.session_state["replay_df"] = replay_df

                st.session_state[
                    "replay_match_id"
                ] = selected_match_id

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

            # =================================================
            # WIN PROBABILITY CHART
            # =================================================

            st.subheader(
                "📊 Win Probability Throughout the Chase"
            )

            chart_df = replay_df[
                [
                    "legal_balls",
                    "win_probability"
                ]
            ].copy()

            chart_df["win_probability"] = (
                chart_df["win_probability"] * 100
            )

            chart_df = chart_df.rename(
                columns={
                    "legal_balls": "Legal Balls",
                    "win_probability":
                        "Batting Team Win Probability (%)"
                }
            )

            st.line_chart(
                chart_df.set_index("Legal Balls"),
                width="stretch"
            )

            # =================================================
            # INTERACTIVE DELIVERY ANALYSIS
            # =================================================

            st.divider()

            st.subheader(
                "🔎 Analyze a Specific Delivery"
            )

            st.markdown(
                """
Select a point in the chase to inspect the exact match
state and understand what influenced the model's prediction.
"""
            )

            # -------------------------------------------------
            # DELIVERY SELECTOR
            # -------------------------------------------------

            delivery_numbers = list(
                range(
                    1,
                    len(replay_df) + 1
                )
            )

            selected_delivery_number = st.slider(
                "Select Delivery State",
                min_value=1,
                max_value=len(replay_df),
                value=len(replay_df),
                step=1
            )

            selected_index = (
                selected_delivery_number - 1
            )

            selected_state = replay_df.iloc[
                selected_index
            ]

            # -------------------------------------------------
            # PREVIOUS STATE
            # -------------------------------------------------

            if selected_index > 0:

                previous_state = replay_df.iloc[
                    selected_index - 1
                ]

                previous_probability = (
                    previous_state[
                        "win_probability"
                    ] * 100
                )

            else:

                previous_state = None
                previous_probability = None

            current_probability = (
                selected_state[
                    "win_probability"
                ] * 100
            )

            # -------------------------------------------------
            # DELIVERY INFORMATION
            # -------------------------------------------------

            st.markdown(
                f"### Delivery {selected_delivery_number}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Over",
                    int(
                        selected_state["over"]
                    )
                )

            with col2:

                st.metric(
                    "Ball",
                    str(
                        selected_state["ball"]
                    )
                )

            with col3:

                st.metric(
                    "Batter",
                    str(
                        selected_state["batter"]
                    )
                )

            with col4:

                st.metric(
                    "Bowler",
                    str(
                        selected_state["bowler"]
                    )
                )

            # -------------------------------------------------
            # DELIVERY RESULT
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Runs From Delivery",
                    int(
                        selected_state["total_runs"]
                    )
                )

            with col2:

                if selected_state["wicket"]:

                    st.error(
                        "❌ WICKET"
                    )

                else:

                    st.success(
                        "No wicket"
                    )

            # =================================================
            # SELECTED WIN PROBABILITY
            # =================================================

            st.subheader(
                "🎯 Win Probability at This Point"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Chasing Team",
                    f"{current_probability:.2f}%"
                )

            with col2:

                if previous_probability is not None:

                    change = (
                        current_probability
                        - previous_probability
                    )

                    st.metric(
                        "Change From Previous Delivery",
                        f"{change:+.2f} pp"
                    )

                else:

                    st.metric(
                        "Change From Previous Delivery",
                        "N/A"
                    )

            with col3:

                st.metric(
                    "Opposition",
                    f"{100 - current_probability:.2f}%"
                )

            # -------------------------------------------------
            # PROBABILITY BAR
            # -------------------------------------------------

            selected_probability_bar = f"""
<div style="
width:100%;
height:24px;
background:#444;
border-radius:12px;
overflow:hidden;
margin-top:15px;
margin-bottom:15px;
">

<div style="
width:{current_probability}%;
height:100%;
background:#ff4b4b;
border-radius:12px 0 0 12px;
">
</div>

</div>
"""

            st.markdown(
                selected_probability_bar,
                unsafe_allow_html=True
            )

            # =================================================
            # MATCH STATE AT DELIVERY
            # =================================================

            st.subheader(
                "📌 Match State at This Delivery"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Runs Scored",
                    int(
                        selected_state["runs_scored"]
                    )
                )

            with col2:

                st.metric(
                    "Runs Required",
                    max(
                        0,
                        int(
                            selected_state[
                                "runs_required"
                            ]
                        )
                    )
                )

            with col3:

                st.metric(
                    "Balls Remaining",
                    int(
                        selected_state[
                            "balls_remaining"
                        ]
                    )
                )

            with col4:

                st.metric(
                    "Wickets in Hand",
                    int(
                        selected_state[
                            "wickets_in_hand"
                        ]
                    )
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Runs Last 12 Balls",
                    int(
                        selected_state[
                            "runs_last_12_balls"
                        ]
                    )
                )

            with col2:

                st.metric(
                    "Runs Last 6 Balls",
                    int(
                        selected_state[
                            "runs_last_6_balls"
                        ]
                    )
                )

            with col3:

                if (
                    selected_state["balls_remaining"]
                    > 0
                ):

                    required_rr = (
                        selected_state[
                            "runs_required"
                        ]
                        /
                        (
                            selected_state[
                                "balls_remaining"
                            ]
                            / 6
                        )
                    )

                else:

                    required_rr = 0

                st.metric(
                    "Required Run Rate",
                    f"{required_rr:.2f}"
                )

            # =================================================
            # DELIVERY IMPACT
            # =================================================

            if previous_probability is not None:

                change = (
                    current_probability
                    - previous_probability
                )

                st.subheader(
                    "⚡ Delivery Impact"
                )

                if change > 5:

                    st.success(
                        f"This delivery increased the "
                        f"model's win probability by "
                        f"{change:.2f} percentage points."
                    )

                elif change > 0:

                    st.info(
                        f"This delivery increased the "
                        f"model's win probability by "
                        f"{change:.2f} percentage points."
                    )

                elif change < -5:

                    st.error(
                        f"This delivery reduced the "
                        f"model's win probability by "
                        f"{abs(change):.2f} percentage points."
                    )

                elif change < 0:

                    st.warning(
                        f"This delivery reduced the "
                        f"model's win probability by "
                        f"{abs(change):.2f} percentage points."
                    )

                else:

                    st.info(
                        "This delivery had almost no "
                        "change in model probability."
                    )

            # =================================================
            # MODEL CONTRIBUTIONS FOR SELECTED STATE
            # =================================================

            st.subheader(
                "🧠 What Influenced the Prediction?"
            )

            # Build feature dictionary exactly as the
            # predictor expects.

            selected_features = (
                predictor.calculate_features(
                    runs_scored=int(
                        selected_state[
                            "runs_scored"
                        ]
                    ),
                    target=int(
                        selected_state[
                            "runs_scored"
                        ]
                        +
                        selected_state[
                            "runs_required"
                        ]
                    ),
                    legal_balls=int(
                        selected_state[
                            "legal_balls"
                        ]
                    ),
                    wickets_lost=int(
                        selected_state[
                            "wickets_lost"
                        ]
                    ),
                    runs_last_12_balls=int(
                        selected_state[
                            "runs_last_12_balls"
                        ]
                    ),
                    runs_last_6_balls=int(
                        selected_state[
                            "runs_last_6_balls"
                        ]
                    ),
                    current_over=int(
                        selected_state[
                            "over"
                        ]
                    )
                )
            )

            selected_contributions = (
                predictor.get_prediction_contributions(
                    selected_features
                )
            )

            positive_selected = (
                selected_contributions[
                    selected_contributions[
                        "contribution"
                    ] > 0
                ]
                .head(5)
                .copy()
            )

            negative_selected = (
                selected_contributions[
                    selected_contributions[
                        "contribution"
                    ] < 0
                ]
                .head(5)
                .copy()
            )

            col1, col2 = st.columns(2)

            # -------------------------------------------------
            # POSITIVE
            # -------------------------------------------------

            with col1:

                st.markdown(
                    "### 📈 Positive Factors"
                )

                positive_display = (
                    positive_selected[
                        [
                            "feature",
                            "value",
                            "contribution"
                        ]
                    ]
                    .copy()
                )

                positive_display[
                    "value"
                ] = positive_display[
                    "value"
                ].round(3)

                positive_display[
                    "contribution"
                ] = positive_display[
                    "contribution"
                ].round(3)

                positive_display = (
                    positive_display.rename(
                        columns={
                            "feature": "Feature",
                            "value": "Current Value",
                            "contribution":
                                "Contribution"
                        }
                    )
                )

                st.dataframe(
                    positive_display,
                    width="stretch",
                    hide_index=True
                )

            # -------------------------------------------------
            # NEGATIVE
            # -------------------------------------------------

            with col2:

                st.markdown(
                    "### 📉 Negative Factors"
                )

                negative_display = (
                    negative_selected[
                        [
                            "feature",
                            "value",
                            "contribution"
                        ]
                    ]
                    .copy()
                )

                negative_display[
                    "value"
                ] = negative_display[
                    "value"
                ].round(3)

                negative_display[
                    "contribution"
                ] = negative_display[
                    "contribution"
                ].round(3)

                negative_display = (
                    negative_display.rename(
                        columns={
                            "feature": "Feature",
                            "value": "Current Value",
                            "contribution":
                                "Contribution"
                        }
                    )
                )

                st.dataframe(
                    negative_display,
                    width="stretch",
                    hide_index=True
                )

            # =================================================
            # LATEST STATE
            # =================================================

            st.divider()

            latest = replay_df.iloc[-1]

            st.subheader(
                "🎯 Final Match State"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Final Runs",
                    int(
                        latest["runs_scored"]
                    )
                )

            with col2:

                st.metric(
                    "Final Wickets",
                    int(
                        latest["wickets_lost"]
                    )
                )

            with col3:

                st.metric(
                    "Balls Used",
                    int(
                        latest["legal_balls"]
                    )
                )

            with col4:

                st.metric(
                    "Final Win Probability",
                    f"{latest['win_probability'] * 100:.2f}%"
                )

            # =================================================
            # FULL DELIVERY TABLE
            # =================================================

            with st.expander(
                "📋 View Delivery-by-Delivery Data"
            ):

                display_replay = replay_df.copy()

                display_replay[
                    "win_probability"
                ] = (
                    display_replay[
                        "win_probability"
                    ] * 100
                ).round(2)

                st.dataframe(
                    display_replay,
                    width="stretch",
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
    # SCORE
    # -----------------------------------------------------

    st.subheader(
        "📌 Current Match State"
    )

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
    # BALLS
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
            max(
                0,
                120 - legal_balls
            )
        )

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    st.subheader(
        "⚡ Recent Batting Momentum"
    )

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

    st.subheader(
        "✅ Input Validation"
    )

    validation_errors = []

    if runs_scored > target:

        validation_errors.append(
            "Runs scored cannot be greater than the target."
        )

    if legal_balls > 120:

        validation_errors.append(
            "Legal balls cannot exceed 120."
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

    predict_button = st.button(
        "🔮 Calculate Win Probability",
        type="primary",
        width="stretch"
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if predict_button and valid_input:

        if "live_result" in st.session_state:

            previous_result = (
                st.session_state["live_result"]
            )

            st.session_state[
                "previous_live_probability"
            ] = (
                previous_result[
                    "batting_team_probability"
                ] * 100
            )

        else:

            st.session_state[
                "previous_live_probability"
            ] = None

        result = predictor.predict(
            runs_scored=runs_scored,
            target=target,
            legal_balls=legal_balls,
            wickets_lost=wickets_lost,
            runs_last_12_balls=runs_last_12_balls,
            runs_last_6_balls=runs_last_6_balls,
            current_over=current_over
        )

        st.session_state[
            "live_result"
        ] = result

        st.session_state[
            "live_batting_team"
        ] = batting_team

        st.session_state[
            "live_opposition"
        ] = opposition

    # =====================================================
    # DISPLAY LIVE RESULT
    # =====================================================

    if "live_result" in st.session_state:

        result = st.session_state[
            "live_result"
        ]

        batting_team = st.session_state.get(
            "live_batting_team",
            batting_team
        )

        opposition = st.session_state.get(
            "live_opposition",
            opposition
        )

        probability = (
            result[
                "batting_team_probability"
            ] * 100
        )

        opposition_probability = (
            result[
                "opposition_probability"
            ] * 100
        )

        previous_probability = (
            st.session_state.get(
                "previous_live_probability"
            )
        )

        # =================================================
        # WIN PROBABILITY
        # =================================================

        st.divider()

        st.subheader(
            "🎯 Win Probability"
        )

        if previous_probability is not None:

            probability_change = (
                probability
                - previous_probability
            )

        else:

            probability_change = 0

        col1, col2, col3 = st.columns(
            [1, 2, 1]
        )

        with col1:

            st.metric(
                batting_team,
                f"{probability:.2f}%"
            )

        with col2:

            probability_card = f"""
<div style="
text-align:center;
padding:25px;
border-radius:15px;
background:rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.10);
">

<div style="
font-size:16px;
opacity:0.7;
">
BATTING TEAM WIN PROBABILITY
</div>

<div style="
font-size:52px;
font-weight:700;
margin-top:8px;
">
{probability:.2f}%
</div>

</div>
"""

            st.markdown(
                probability_card,
                unsafe_allow_html=True
            )

        with col3:

            st.metric(
                opposition,
                f"{opposition_probability:.2f}%"
            )

        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        probability_bar = f"""
<div style="
width:100%;
height:24px;
background:#444;
border-radius:12px;
overflow:hidden;
margin-top:20px;
margin-bottom:8px;
">

<div style="
width:{probability}%;
height:100%;
background:#ff4b4b;
border-radius:12px 0 0 12px;
">
</div>

</div>
"""

        st.markdown(
            probability_bar,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # CHANGE
        # -------------------------------------------------

        if previous_probability is not None:

            if probability_change > 0:

                st.success(
                    f"📈 Win probability increased by "
                    f"{probability_change:.2f} percentage points."
                )

            elif probability_change < 0:

                st.warning(
                    f"📉 Win probability decreased by "
                    f"{abs(probability_change):.2f} percentage points."
                )

            else:

                st.info(
                    "➡️ Win probability is unchanged."
                )

        else:

            st.info(
                "ℹ️ This is the first prediction."
            )

        # =================================================
        # MATCH SITUATION
        # =================================================

        st.subheader(
            "📌 Match Situation"
        )

        if probability >= 75:

            situation = "🟢 Strong position"

        elif probability >= 60:

            situation = "🟢 Advantage batting team"

        elif probability >= 50:

            situation = "🟡 Slight advantage batting team"

        elif probability >= 40:

            situation = "🟡 Slight advantage opposition"

        elif probability >= 25:

            situation = "🟠 Advantage opposition"

        else:

            situation = "🔴 Difficult position"

        st.markdown(
            f"### {situation}"
        )

        # =================================================
        # MATCH STATE
        # =================================================

        st.subheader(
            "📊 Match State"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Runs Required",
                int(
                    result["runs_required"]
                )
            )

        with col2:

            st.metric(
                "Balls Remaining",
                int(
                    result["balls_remaining"]
                )
            )

        with col3:

            st.metric(
                "Wickets in Hand",
                int(
                    result["wickets_in_hand"]
                )
            )

        with col4:

            st.metric(
                "Phase",
                result["phase"]
            )

        # =================================================
        # ANALYTICS
        # =================================================

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
                int(
                    result["runs_required"]
                )
            )

        # =================================================
        # INTERPRETATION
        # =================================================

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
                f"The match is relatively balanced, "
                f"with {batting_team} at "
                f"{probability:.2f}%."
            )

        else:

            st.warning(
                f"The model currently estimates a "
                f"{probability:.2f}% win probability "
                f"for {batting_team}."
            )

        # =================================================
        # MODEL FEATURES
        # =================================================

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
                        str(
                            result["runs_scored"]
                        ),
                        str(
                            result["runs_required"]
                        ),
                        str(
                            result["balls_remaining"]
                        ),
                        str(
                            result["wickets_in_hand"]
                        ),
                        f"{result['current_run_rate']:.3f}",
                        f"{result['required_run_rate']:.3f}",
                        str(
                            result[
                                "runs_last_12_balls"
                            ]
                        ),
                        str(
                            result[
                                "runs_last_6_balls"
                            ]
                        ),
                        str(
                            result["phase"]
                        )
                    ]
                }
            )

            st.dataframe(
                feature_display,
                width="stretch",
                hide_index=True
            )

        # =================================================
        # MODEL CONTRIBUTIONS
        # =================================================

        st.subheader(
            "🧠 Why Did the Model Make This Prediction?"
        )

        st.caption(
            "Positive values push the prediction toward "
            "the batting team; negative values push it away. "
            "These are model contribution values, not "
            "percentage points."
        )

        contribution_df = (
            predictor.get_prediction_contributions(
                result
            )
        )

        positive_contributions = (
            contribution_df[
                contribution_df[
                    "contribution"
                ] > 0
            ].copy()
        )

        negative_contributions = (
            contribution_df[
                contribution_df[
                    "contribution"
                ] < 0
            ].copy()
        )

        col1, col2 = st.columns(2)

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

            positive_display[
                "value"
            ] = positive_display[
                "value"
            ].round(3)

            positive_display[
                "contribution"
            ] = positive_display[
                "contribution"
            ].round(3)

            positive_display = (
                positive_display.rename(
                    columns={
                        "feature": "Feature",
                        "value": "Current Value",
                        "contribution":
                            "Contribution"
                    }
                )
            )

            st.dataframe(
                positive_display,
                width="stretch",
                hide_index=True
            )

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

            negative_display[
                "value"
            ] = negative_display[
                "value"
            ].round(3)

            negative_display[
                "contribution"
            ] = negative_display[
                "contribution"
            ].round(3)

            negative_display = (
                negative_display.rename(
                    columns={
                        "feature": "Feature",
                        "value": "Current Value",
                        "contribution":
                            "Contribution"
                    }
                )
            )

            st.dataframe(
                negative_display,
                width="stretch",
                hide_index=True
            )

        # =================================================
        # GLOBAL MODEL EFFECTS
        # =================================================

        with st.expander(
            "📚 View Global Model Feature Effects"
        ):

            coefficients = (
                predictor.explain_prediction()
            )

            coefficients[
                "coefficient"
            ] = coefficients[
                "coefficient"
            ].round(4)

            coefficients = (
                coefficients.rename(
                    columns={
                        "feature": "Feature",
                        "coefficient":
                            "Coefficient",
                        "effect": "Effect"
                    }
                )
            )

            st.dataframe(
                coefficients[
                    [
                        "Feature",
                        "Coefficient",
                        "Effect"
                    ]
                ],
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Global coefficients describe the Logistic "
                "Regression model's learned direction of "
                "association. They are not the same as the "
                "contribution of a feature for one specific "
                "match state."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "WPL Cricket Win Probability & Match Analytics Engine | "
    "Model-derived probabilities for analytical purposes"
)