import os
import sys

import pandas as pd
import streamlit as st

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.predictor import WPLPredictor
from src.replay import MatchReplay


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="WPL Win Probability Engine",
    page_icon="🏏",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL + DATA
# --------------------------------------------------

@st.cache_resource
def load_predictor():
    return WPLPredictor()


@st.cache_resource
def load_replay():
    return MatchReplay()


predictor = load_predictor()
replay = load_replay()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏏 WPL Win Probability Engine")

st.markdown(
    """
    **Machine Learning powered cricket match analytics**

    Predicting win probability from live match state using
    historical Women's Premier League data.
    """
)

st.divider()


# ==================================================
# SECTION 1 — HISTORICAL MATCH REPLAY
# ==================================================

st.header("📈 Historical Match Replay")

st.write(
    "Select a WPL match to replay its second innings "
    "delivery-by-delivery and visualize how the predicted "
    "win probability changes."
)


# --------------------------------------------------
# GET MATCH LIST
# --------------------------------------------------

matches = (
    replay.df[
        [
            "match_id",
            "date",
            "team1",
            "team2",
            "winner"
        ]
    ]
    .drop_duplicates("match_id")
    .sort_values("date")
    .reset_index(drop=True)
)


# Only matches with a recorded winner
matches = matches[
    matches["winner"].notna()
].copy()


# --------------------------------------------------
# MATCH SELECTOR
# --------------------------------------------------

match_options = []

for _, row in matches.iterrows():

    label = (
        f"{row['team1']} vs {row['team2']} "
        f"| {row['date']} "
        f"| Match ID: {row['match_id']}"
    )

    match_options.append(label)


selected_match = st.selectbox(
    "Select WPL Match",
    match_options
)


# Get selected match ID
selected_index = match_options.index(selected_match)

selected_match_id = matches.iloc[
    selected_index
]["match_id"]


# --------------------------------------------------
# MATCH DETAILS
# --------------------------------------------------

selected_info = matches[
    matches["match_id"] == selected_match_id
].iloc[0]


team1 = selected_info["team1"]
team2 = selected_info["team2"]
winner = selected_info["winner"]


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Teams",
        f"{team1} vs {team2}"
    )

with col2:
    st.metric(
        "Winner",
        winner
    )

with col3:
    st.metric(
        "Match ID",
        str(selected_match_id)
    )


# --------------------------------------------------
# RUN REPLAY
# --------------------------------------------------

if st.button(
    "▶ Replay Match",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Replaying match and calculating win probabilities..."
    ):

        replay_data = replay.replay_second_innings(
            selected_match_id
        )


    # Store replay data in session state
    st.session_state["replay_data"] = replay_data
    st.session_state["replay_match_id"] = selected_match_id


# --------------------------------------------------
# DISPLAY REPLAY
# --------------------------------------------------

if (
    "replay_data" in st.session_state
    and
    st.session_state.get("replay_match_id")
    == selected_match_id
):

    replay_data = st.session_state["replay_data"]

    if replay_data.empty:

        st.warning(
            "No prediction states were generated for this match."
        )

    else:

        st.divider()

        st.subheader(
            "Win Probability Progression"
        )


        # --------------------------------------------------
        # PREPARE CHART DATA
        # --------------------------------------------------

        chart_data = replay_data[
            [
                "legal_balls",
                "win_probability"
            ]
        ].copy()

        chart_data["Win Probability (%)"] = (
            chart_data["win_probability"] * 100
        )

        chart_data = chart_data.rename(
            columns={
                "legal_balls": "Legal Balls"
            }
        )


        # --------------------------------------------------
        # CHART
        # --------------------------------------------------

        st.line_chart(
            chart_data.set_index("Legal Balls")[
                "Win Probability (%)"
            ],
            height=450
        )


        # --------------------------------------------------
        # LATEST MATCH STATE
        # --------------------------------------------------

        latest = replay_data.iloc[-1]


        st.subheader(
            "Latest Match State"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Score",
                f"{int(latest['runs_scored'])}/"
                f"{int(latest['wickets_lost'])}"
            )

        with col2:
            st.metric(
                "Runs Required",
                int(latest["runs_required"])
            )

        with col3:
            st.metric(
                "Balls Remaining",
                int(latest["balls_remaining"])
            )

        with col4:
            st.metric(
                "Wickets in Hand",
                10 - int(latest["wickets_lost"])
            )


        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        st.subheader(
            "Current Win Probability"
        )

        batting_probability = (
            latest["win_probability"]
        )

        opposition_probability = (
            1 - batting_probability
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Chasing Team",
                f"{batting_probability * 100:.2f}%"
            )

            st.progress(
                float(batting_probability)
            )

        with col2:

            st.metric(
                "Defending Team",
                f"{opposition_probability * 100:.2f}%"
            )

            st.progress(
                float(opposition_probability)
            )


        # --------------------------------------------------
        # ANALYTICS
        # --------------------------------------------------

        st.subheader(
            "Match Analytics"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current RR",
                f"{latest['current_run_rate']:.2f}"
            )

        with col2:
            st.metric(
                "Required RR",
                f"{latest['required_run_rate']:.2f}"
            )

        with col3:
            st.metric(
                "Last 12 Balls",
                int(latest["runs_last_12_balls"])
            )

        with col4:
            st.metric(
                "Phase",
                latest["phase"]
            )


        # --------------------------------------------------
        # DATA TABLE
        # --------------------------------------------------

        with st.expander(
            "View Delivery-by-Delivery Data"
        ):

            display_data = replay_data.copy()

            display_data[
                "win_probability"
            ] = (
                display_data["win_probability"] * 100
            ).round(2)

            display_data = display_data.rename(
                columns={
                    "legal_balls": "Legal Balls",
                    "over": "Over",
                    "ball": "Ball",
                    "runs_scored": "Score",
                    "wickets_lost": "Wickets",
                    "runs_required": "Runs Required",
                    "balls_remaining": "Balls Remaining",
                    "current_run_rate": "Current RR",
                    "required_run_rate": "Required RR",
                    "runs_last_12_balls": "Last 12 Balls",
                    "runs_last_6_balls": "Last 6 Balls",
                    "phase": "Phase",
                    "win_probability": "Win Probability (%)"
                }
            )

            st.dataframe(
                display_data[
                    [
                        "Legal Balls",
                        "Over",
                        "Ball",
                        "Score",
                        "Wickets",
                        "Runs Required",
                        "Balls Remaining",
                        "Current RR",
                        "Required RR",
                        "Last 12 Balls",
                        "Last 6 Balls",
                        "Phase",
                        "Win Probability (%)"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )