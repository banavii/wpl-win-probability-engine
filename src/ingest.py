import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. FIND ALL WPL JSON FILES
# --------------------------------------------------

data_folder = Path("data/raw/wpl/wpl_json")

json_files = list(data_folder.glob("*.json"))

print("Number of JSON files found:", len(json_files))


# --------------------------------------------------
# 2. STORE ALL DELIVERIES
# --------------------------------------------------

all_deliveries = []


# --------------------------------------------------
# 3. PROCESS EACH MATCH
# --------------------------------------------------

for file_path in json_files:

    print("Processing:", file_path.name)

    # Read JSON
    with open(file_path, "r") as file:
        match = json.load(file)


    # --------------------------------------------------
    # MATCH INFORMATION
    # --------------------------------------------------

    info = match["info"]

    # Get match ID from filename
    match_id = file_path.stem

    season = info["season"]
    dates = info["dates"]
    venue = info.get("venue")
    city = info.get("city")

    teams = info["teams"]

    team1 = teams[0]
    team2 = teams[1]

    # Winner
    winner = info.get("outcome", {}).get("winner")

    # Toss
    toss = info.get("toss", {})

    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")


    # --------------------------------------------------
    # PROCESS INNINGS
    # --------------------------------------------------

    for innings_number, innings in enumerate(
        match["innings"],
        start=1
    ):

        batting_team = innings["team"]


        # --------------------------------------------------
        # PROCESS OVERS
        # --------------------------------------------------

        for over in innings["overs"]:

            over_number = over["over"]


            # --------------------------------------------------
            # PROCESS DELIVERIES
            # --------------------------------------------------

            for ball_number, delivery in enumerate(
                over["deliveries"],
                start=1
            ):

                # Runs
                runs = delivery["runs"]

                batter_runs = runs["batter"]
                extras = runs["extras"]
                total_runs = runs["total"]


                # Extras
                extra_type = None

                if "extras" in delivery:

                    extra_types = list(
                        delivery["extras"].keys()
                    )

                    if extra_types:
                        extra_type = extra_types[0]


                # Wicket
                player_out = None
                wicket_type = None

                if "wickets" in delivery:

                    wicket = delivery["wickets"][0]

                    player_out = wicket.get(
                        "player_out"
                    )

                    wicket_type = wicket.get(
                        "kind"
                    )


                # Create row
                row = {

                    # Match information
                    "match_id": match_id,
                    "season": season,
                    "date": dates[0],
                    "venue": venue,
                    "city": city,

                    "team1": team1,
                    "team2": team2,

                    "winner": winner,

                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,


                    # Innings information
                    "innings": innings_number,
                    "batting_team": batting_team,

                    "over": over_number,
                    "ball": ball_number,


                    # Players
                    "batter": delivery["batter"],
                    "bowler": delivery["bowler"],
                    "non_striker": delivery["non_striker"],


                    # Runs
                    "batter_runs": batter_runs,
                    "extras": extras,
                    "total_runs": total_runs,


                    # Extras
                    "extra_type": extra_type,


                    # Wicket
                    "player_out": player_out,
                    "wicket_type": wicket_type
                }


                all_deliveries.append(row)


# --------------------------------------------------
# 4. CREATE ONE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(all_deliveries)


# --------------------------------------------------
# 5. BASIC CHECKS
# --------------------------------------------------

print("\n--------------------------------")
print("INGESTION COMPLETE")
print("--------------------------------")

print("Matches processed:", df["match_id"].nunique())

print("Total deliveries:", len(df))

print("Total columns:", len(df.columns))


# --------------------------------------------------
# 6. SHOW SAMPLE
# --------------------------------------------------

print("\nFirst 5 rows:")

print(
    df.head().to_string()
)


# --------------------------------------------------
# 7. SAVE PROCESSED DATA
# --------------------------------------------------

output_folder = Path("data/processed")

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


output_file = output_folder / "deliveries.parquet"

df.to_parquet(
    output_file,
    index=False
)


print("\nSaved to:")

print(output_file)