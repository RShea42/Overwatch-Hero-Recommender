import tomllib

import pandas as pd
import requests

MATCHUPS_URL = (
    "https://raw.githubusercontent.com/MaikBuse/minmax-watch/main/data/matchups.toml"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

REQUEST_TIMEOUT_SECONDS = 10

OUTPUT_PATH = "data/matchups.csv"


def fetch_matchups_toml():
    response = requests.get(MATCHUPS_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return tomllib.loads(response.text)


def parse_records(data):
    records = []
    for entry in data["matchup"]:
        curated = entry.get("curated")
        value = entry["value"]
        resolved_value = curated if curated is not None else value
        records.append(
            {
                "hero_id": entry["hero"],
                "opponent_id": entry["vs"],
                "value": value,
                "curated": curated,
                "resolved_value": resolved_value,
                "cpgg": entry.get("cpgg"),
                "opick": entry.get("opick"),
                "cwatch": entry.get("cwatch"),
                "disagreement": entry.get("disagreement", False),
            }
        )
    return records


def print_validation(df):
    print(f"Total matchup records: {len(df)}")
    print(f"Unique hero IDs: {df['hero_id'].nunique()}")
    print(f"Unique opponent IDs: {df['opponent_id'].nunique()}")

    dupe_sizes = df.groupby(["hero_id", "opponent_id"]).size()
    duplicate_pairs = int((dupe_sizes > 1).sum())
    print(f"Duplicate hero_id+opponent_id pairs: {duplicate_pairs}")

    print(f"Minimum resolved_value: {df['resolved_value'].min()}")
    print(f"Maximum resolved_value: {df['resolved_value'].max()}")

    print(f"Records using curated overrides: {df['curated'].notna().sum()}")
    print(f"Records marked disagreement: {int(df['disagreement'].sum())}")

    print(f"Missing cpgg values: {df['cpgg'].isna().sum()}")
    print(f"Missing opick values: {df['opick'].isna().sum()}")
    print(f"Missing cwatch values: {df['cwatch'].isna().sum()}")

    has_soldier76 = (df["hero_id"] == "soldier-76").any()
    print(f"Soldier: 76 present as hero: {has_soldier76}")

    if has_soldier76:
        soldier76 = df[df["hero_id"] == "soldier-76"]

        worst = soldier76.sort_values("resolved_value", ascending=True).head(5)
        best = soldier76.sort_values("resolved_value", ascending=False).head(5)

        print("\nSoldier: 76 - five worst matchups (by resolved_value):")
        print(worst[["opponent_id", "resolved_value"]].to_string(index=False))

        print("\nSoldier: 76 - five best matchups (by resolved_value):")
        print(best[["opponent_id", "resolved_value"]].to_string(index=False))


def main():
    data = fetch_matchups_toml()
    records = parse_records(data)

    df = pd.DataFrame(
        records,
        columns=[
            "hero_id",
            "opponent_id",
            "value",
            "curated",
            "resolved_value",
            "cpgg",
            "opick",
            "cwatch",
            "disagreement",
        ],
    )
    df.to_csv(OUTPUT_PATH, index=False)

    print_validation(df)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
