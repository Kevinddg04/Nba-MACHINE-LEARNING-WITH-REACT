import pandas as pd
import json

CSV_PATH = "TeamStatistics.csv"

try:
    df = pd.read_csv(CSV_PATH)
    unique_teams = df["teamId"].unique()
    print(f"Total unique teams: {len(unique_teams)}")
    print(f"Is Oklahoma City Thunder (1610612760) in the dataset?: {1610612760 in unique_teams}")
    recent = df[df["teamId"] == 1610612760].tail(1)
    if not recent.empty:
        print("Latest OKC game:", recent["gameDateTimeEst"].values[0])
    
    # get active teams in 2024
    if "gameDateTimeEst" in df.columns:
        df["year"] = pd.to_datetime(df["gameDateTimeEst"]).dt.year
        teams_2024 = df[df["year"] == 2024]["teamId"].unique()
        print(f"Teams with games in 2024: {len(teams_2024)}")
except Exception as e:
    print("Error:", e)
