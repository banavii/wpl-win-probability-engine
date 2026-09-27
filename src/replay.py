import pandas as pd

from src.predictor import WPLPredictor


class MatchReplay:

    def __init__(
        self,
        data_path="data/processed/deliveries.parquet",
        model_path="models/wpl_win_probability_model.joblib"
    ):
        # Load processed delivery data
        self.df = pd.read_parquet(data_path)

        # Load trained ML model
        self.predictor = WPLPredictor(model_path)

    # --------------------------------------------------
    # GET MATCH DATA
    # --------------------------------------------------

    def get_match(self, match_id):
        """
        Return all deliveries belonging to one match.
        """

        match_df = self.df[
            self.df["match_id"].astype(str) == str(match_id)
        ].copy()

        if match_df.empty:
            raise ValueError(
                f"Match {match_id} not found."
            )

        return match_df.sort_values(
            ["innings", "over", "ball"]
        ).reset_index(drop=True)

    # --------------------------------------------------
    # REPLAY SECOND INNINGS
    # --------------------------------------------------

    def replay_second_innings(self, match_id):
        """
        Replay the second innings delivery by delivery.

        After every legal delivery, calculate the batting
        team's predicted win probability.
        """

        # Get match
        match_df = self.get_match(match_id)

        # Get second innings
        innings_df = match_df[
            match_df["innings"] == 2
        ].copy()

        if innings_df.empty:
            raise ValueError(
                f"Match {match_id} does not contain a second innings."
            )

        # --------------------------------------------------
        # CALCULATE TARGET
        # --------------------------------------------------

        first_innings = match_df[
            match_df["innings"] == 1
        ]

        first_innings_score = int(
            first_innings["total_runs"].sum()
        )

        target = first_innings_score + 1

        # --------------------------------------------------
        # INITIAL MATCH STATE
        # --------------------------------------------------

        runs_scored = 0
        wickets_lost = 0
        legal_balls = 0

        # Store runs from legal deliveries
        legal_delivery_history = []

        # Store probability after every legal delivery
        probability_history = []

        # --------------------------------------------------
        # REPLAY EACH DELIVERY
        # --------------------------------------------------

        for _, delivery in innings_df.iterrows():

            # ----------------------------------------------
            # ADD RUNS
            # ----------------------------------------------

            runs_scored += int(
                delivery["total_runs"]
            )

            # ----------------------------------------------
            # CHECK LEGAL DELIVERY
            # ----------------------------------------------

            # Wides and no-balls do not count as legal balls.
            is_legal = delivery["extra_type"] not in [
                "wides",
                "noballs"
            ]

            # ----------------------------------------------
            # CHECK WICKET
            # ----------------------------------------------

            wicket = pd.notna(
                delivery["player_out"]
            )

            if wicket:
                wickets_lost += 1

            # ----------------------------------------------
            # PROCESS LEGAL DELIVERY
            # ----------------------------------------------

            if is_legal:

                legal_balls += 1

                # Store runs from this legal delivery
                legal_delivery_history.append(
                    int(delivery["total_runs"])
                )

                # ------------------------------------------
                # RECENT MOMENTUM
                # ------------------------------------------

                runs_last_12_balls = sum(
                    legal_delivery_history[-12:]
                )

                runs_last_6_balls = sum(
                    legal_delivery_history[-6:]
                )

                # ------------------------------------------
                # CURRENT OVER
                # ------------------------------------------

                current_over = int(
                    delivery["over"]
                )

                # ------------------------------------------
                # RUNS REQUIRED
                # ------------------------------------------

                runs_required = target - runs_scored

                # ------------------------------------------
                # CHECK WHETHER MODEL CAN PREDICT
                # ------------------------------------------

                can_predict = (
                    runs_required > 0
                    and legal_balls < 120
                    and wickets_lost < 10
                )

                if can_predict:

                    # --------------------------------------
                    # MODEL PREDICTION
                    # --------------------------------------

                    result = self.predictor.predict(
                        runs_scored=runs_scored,
                        target=target,
                        legal_balls=legal_balls,
                        wickets_lost=wickets_lost,
                        runs_last_12_balls=runs_last_12_balls,
                        runs_last_6_balls=runs_last_6_balls,
                        current_over=current_over
                    )

                    probability = result[
                        "batting_team_probability"
                    ]

                    # --------------------------------------
                    # SAVE MATCH STATE
                    # --------------------------------------

                    probability_history.append({

                        "match_id": match_id,

                        "innings": 2,

                        "legal_balls": legal_balls,

                        "over": current_over,

                        "ball": delivery["ball"],

                        "runs_scored": runs_scored,

                        "wickets_lost": wickets_lost,

                        "runs_required": runs_required,

                        "balls_remaining": max(
                            0,
                            120 - legal_balls
                        ),

                        "current_run_rate": result[
                            "current_run_rate"
                        ],

                        "required_run_rate": result[
                            "required_run_rate"
                        ],

                        "runs_last_12_balls":
                            runs_last_12_balls,

                        "runs_last_6_balls":
                            runs_last_6_balls,

                        "phase": result["phase"],

                        "win_probability": probability

                    })

                # ------------------------------------------
                # STOP CONDITIONS
                # ------------------------------------------

                # Chasing team has won
                if runs_scored >= target:
                    break

                # Full 20 overs completed
                if legal_balls >= 120:
                    break

                # All 10 wickets lost
                if wickets_lost >= 10:
                    break

        # --------------------------------------------------
        # RETURN RESULTS
        # --------------------------------------------------

        return pd.DataFrame(
            probability_history
        )