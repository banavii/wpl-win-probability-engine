import pandas as pd

from src.predictor import WPLPredictor


DATA_PATH = "data/processed/deliveries.parquet"


class WPLReplay:

    def __init__(self):
        self.df = pd.read_parquet(DATA_PATH)
        self.predictor = WPLPredictor()

    def get_match(self, match_id):
        match_df = self.df[
            self.df["match_id"] == match_id
        ].copy()

        return match_df

    def replay_second_innings(self, match_id):

        match_df = self.get_match(match_id)

        if match_df.empty:
            raise ValueError(
                f"Match {match_id} not found."
            )

        # -------------------------------------------------
        # FIRST INNINGS
        # -------------------------------------------------

        first_innings = match_df[
            match_df["innings"] == 1
        ]

        target = (
            first_innings["total_runs"].sum()
            + 1
        )

        # -------------------------------------------------
        # SECOND INNINGS
        # -------------------------------------------------

        second_innings = match_df[
            match_df["innings"] == 2
        ].copy()

        if second_innings.empty:
            raise ValueError(
                "Second innings not found."
            )

        # -------------------------------------------------
        # REPLAY VARIABLES
        # -------------------------------------------------

        runs_scored = 0
        wickets_lost = 0
        legal_balls = 0

        recent_legal_runs = []

        replay_rows = []

        # -------------------------------------------------
        # DELIVERY LOOP
        # -------------------------------------------------

        for _, delivery in second_innings.iterrows():

            # Determine whether delivery is legal
            is_legal = (
                delivery["extra_type"]
                not in ["wides", "noballs"]
            )

            # Add runs
            runs_scored += int(
                delivery["total_runs"]
            )

            # Add wicket
            if pd.notna(delivery["player_out"]):
                wickets_lost += 1

            # -------------------------------------------------
            # ONLY UPDATE BALL-BASED FEATURES
            # FOR LEGAL DELIVERIES
            # -------------------------------------------------

            if is_legal:

                legal_balls += 1

                recent_legal_runs.append(
                    int(delivery["total_runs"])
                )

                # Keep only last 12 legal balls
                if len(recent_legal_runs) > 12:
                    recent_legal_runs.pop(0)

            # -------------------------------------------------
            # STOP CONDITIONS
            # -------------------------------------------------

            runs_required = target - runs_scored

            balls_remaining = max(
                0,
                120 - legal_balls
            )

            wickets_in_hand = max(
                0,
                10 - wickets_lost
            )

            # If match is already won
            if runs_scored >= target:
                runs_required = 0

            # -------------------------------------------------
            # CURRENT OVER
            # -------------------------------------------------

            current_over = int(
                delivery["over"]
            )

            # -------------------------------------------------
            # RECENT MOMENTUM
            # -------------------------------------------------

            runs_last_12_balls = sum(
                recent_legal_runs[-12:]
            )

            runs_last_6_balls = sum(
                recent_legal_runs[-6:]
            )

            # -------------------------------------------------
            # ONLY PREDICT AFTER LEGAL DELIVERY
            # -------------------------------------------------

            if is_legal:

                # Match won
                if runs_scored >= target:

                    win_probability = 1.0

                # All wickets lost
                elif wickets_lost >= 10:

                    win_probability = 0.0

                # Innings finished
                elif legal_balls >= 120:

                    win_probability = (
                        1.0
                        if runs_scored >= target
                        else 0.0
                    )

                else:

                    result = self.predictor.predict(
                        runs_scored=runs_scored,
                        target=target,
                        legal_balls=legal_balls,
                        wickets_lost=wickets_lost,
                        runs_last_12_balls=runs_last_12_balls,
                        runs_last_6_balls=runs_last_6_balls,
                        current_over=current_over
                    )

                    win_probability = result[
                        "batting_team_probability"
                    ]

                # -------------------------------------------------
                # SAVE STATE
                # -------------------------------------------------

                replay_rows.append(
                    {
                        "match_id": match_id,
                        "over": current_over,
                        "ball": delivery["ball"],
                        "legal_balls": legal_balls,
                        "runs_scored": runs_scored,
                        "runs_required": max(
                            0,
                            runs_required
                        ),
                        "balls_remaining": balls_remaining,
                        "wickets_lost": wickets_lost,
                        "wickets_in_hand": wickets_in_hand,
                        "runs_last_12_balls": runs_last_12_balls,
                        "runs_last_6_balls": runs_last_6_balls,
                        "win_probability": win_probability,
                        "batter": delivery["batter"],
                        "bowler": delivery["bowler"],
                        "total_runs": delivery["total_runs"],
                        "wicket": pd.notna(
                            delivery["player_out"]
                        )
                    }
                )

            # -------------------------------------------------
            # STOP REPLAY AFTER MATCH ENDS
            # -------------------------------------------------

            if runs_scored >= target:
                break

            if wickets_lost >= 10:
                break

            if legal_balls >= 120:
                break

        # -----------------------------------------------------
        # RETURN DATAFRAME
        # -----------------------------------------------------

        return pd.DataFrame(replay_rows)


# =========================================================
# DASHBOARD-FRIENDLY FUNCTION
# =========================================================

def replay_second_innings(match_id):

    replay = WPLReplay()

    return replay.replay_second_innings(
        match_id
    )