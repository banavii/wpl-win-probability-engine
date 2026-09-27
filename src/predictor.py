import joblib
import pandas as pd


MODEL_PATH = "models/wpl_win_probability_model.joblib"


class WPLPredictor:

    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def calculate_features(
        self,
        runs_scored,
        target,
        legal_balls,
        wickets_lost,
        runs_last_12_balls,
        runs_last_6_balls,
        current_over
    ):
        runs_required = target - runs_scored

        balls_remaining = max(0, 120 - legal_balls)

        wickets_in_hand = max(0, 10 - wickets_lost)

        if legal_balls > 0:
            current_run_rate = runs_scored / (legal_balls / 6)
        else:
            current_run_rate = 0

        if balls_remaining > 0 and runs_required > 0:
            required_run_rate = runs_required / (balls_remaining / 6)
        else:
            required_run_rate = 0

        if current_over <= 5:
            phase = "Powerplay"
        elif current_over <= 14:
            phase = "Middle"
        else:
            phase = "Death"

        return {
            "runs_scored": runs_scored,
            "runs_required": runs_required,
            "balls_remaining": balls_remaining,
            "wickets_in_hand": wickets_in_hand,
            "current_run_rate": current_run_rate,
            "required_run_rate": required_run_rate,
            "runs_last_12_balls": runs_last_12_balls,
            "runs_last_6_balls": runs_last_6_balls,
            "phase": phase
        }

    def predict(
        self,
        runs_scored,
        target,
        legal_balls,
        wickets_lost,
        runs_last_12_balls,
        runs_last_6_balls,
        current_over
    ):
        features = self.calculate_features(
            runs_scored,
            target,
            legal_balls,
            wickets_lost,
            runs_last_12_balls,
            runs_last_6_balls,
            current_over
        )

        input_data = pd.DataFrame([features])

        probability = self.model.predict_proba(input_data)[0][1]

        return {
            **features,
            "batting_team_probability": probability,
            "opposition_probability": 1 - probability
        }