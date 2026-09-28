import joblib
import pandas as pd


MODEL_PATH = "models/wpl_win_probability_model.joblib"


class WPLPredictor:

    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    # --------------------------------------------------
    # FEATURE CALCULATION
    # --------------------------------------------------

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

        balls_remaining = max(
            0,
            120 - legal_balls
        )

        wickets_in_hand = max(
            0,
            10 - wickets_lost
        )

        if legal_balls > 0:
            current_run_rate = (
                runs_scored / (legal_balls / 6)
            )
        else:
            current_run_rate = 0

        if (
            balls_remaining > 0
            and runs_required > 0
        ):
            required_run_rate = (
                runs_required /
                (balls_remaining / 6)
            )
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

    # --------------------------------------------------
    # WIN PROBABILITY
    # --------------------------------------------------

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
            runs_scored=runs_scored,
            target=target,
            legal_balls=legal_balls,
            wickets_lost=wickets_lost,
            runs_last_12_balls=runs_last_12_balls,
            runs_last_6_balls=runs_last_6_balls,
            current_over=current_over
        )

        input_data = pd.DataFrame([features])

        probability = (
            self.model.predict_proba(input_data)[0][1]
        )

        return {
            **features,
            "batting_team_probability": probability,
            "opposition_probability": 1 - probability
        }

    # --------------------------------------------------
    # GLOBAL MODEL COEFFICIENTS
    # --------------------------------------------------

    def get_feature_coefficients(self):

        preprocessor = (
            self.model.named_steps["preprocessor"]
        )

        classifier = (
            self.model.named_steps["classifier"]
        )

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        coefficients = classifier.coef_[0]

        coefficient_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        })

        coefficient_df["abs_coefficient"] = (
            coefficient_df["coefficient"].abs()
        )

        return (
            coefficient_df
            .sort_values(
                "abs_coefficient",
                ascending=False
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # CLEAN FEATURE NAME
    # --------------------------------------------------

    def clean_feature_name(self, feature_name):

        if feature_name.startswith(
            "phase__phase_"
        ):

            return (
                "Phase: "
                + feature_name
                .replace(
                    "phase__phase_",
                    ""
                )
                .title()
            )

        return (
            feature_name
            .replace(
                "numeric__",
                ""
            )
            .replace(
                "_",
                " "
            )
            .title()
        )

    # --------------------------------------------------
    # GLOBAL EXPLANATION
    # --------------------------------------------------

    def explain_prediction(self):

        coefficient_df = (
            self.get_feature_coefficients()
        )

        explanations = []

        for _, row in coefficient_df.iterrows():

            feature_name = row["feature"]
            coefficient = row["coefficient"]

            clean_name = (
                self.clean_feature_name(
                    feature_name
                )
            )

            if coefficient > 0:
                effect = "Positive"
            else:
                effect = "Negative"

            explanations.append({
                "feature": clean_name,
                "coefficient": coefficient,
                "effect": effect
            })

        return pd.DataFrame(
            explanations
        )

    # --------------------------------------------------
    # MATCH-SPECIFIC CONTRIBUTIONS
    # --------------------------------------------------

    def get_prediction_contributions(
        self,
        features
    ):
        """
        Calculate the contribution of each transformed
        feature for the current match state.

        Contribution is calculated in the Logistic
        Regression linear predictor space:

            contribution =
                transformed_value × coefficient

        These values are NOT percentage points.
        """

        # Convert current features to DataFrame
        input_data = pd.DataFrame([features])

        # Get preprocessing stage
        preprocessor = (
            self.model.named_steps["preprocessor"]
        )

        # Get Logistic Regression classifier
        classifier = (
            self.model.named_steps["classifier"]
        )

        # Transform the current match state
        transformed_data = (
            preprocessor.transform(
                input_data
            )
        )

        # Convert sparse matrix if necessary
        if hasattr(
            transformed_data,
            "toarray"
        ):
            transformed_data = (
                transformed_data.toarray()
            )

        # Feature names after preprocessing
        feature_names = (
            preprocessor.get_feature_names_out()
        )

        # Logistic Regression coefficients
        coefficients = classifier.coef_[0]

        # Actual feature values
        values = transformed_data[0]

        # Calculate contribution
        contributions = (
            values * coefficients
        )

        contribution_df = pd.DataFrame({
            "feature": feature_names,
            "value": values,
            "coefficient": coefficients,
            "contribution": contributions
        })

        # Absolute contribution
        contribution_df[
            "abs_contribution"
        ] = (
            contribution_df[
                "contribution"
            ].abs()
        )

        # Clean names
        contribution_df[
            "feature"
        ] = contribution_df[
            "feature"
        ].apply(
            self.clean_feature_name
        )

        # Effect direction
        contribution_df["effect"] = (
            contribution_df[
                "contribution"
            ].apply(
                lambda x:
                    "Positive"
                    if x > 0
                    else "Negative"
            )
        )

        return (
            contribution_df
            .sort_values(
                "abs_contribution",
                ascending=False
            )
            .reset_index(drop=True)
        )