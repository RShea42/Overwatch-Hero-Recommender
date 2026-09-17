import itertools
import time
from collections import Counter

import pandas as pd
import requests

RATES_URL = "https://overwatch.blizzard.com/en-us/rates/data/"

HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

RANKS = [
    "Bronze",
    "Silver",
    "Gold",
    "Platinum",
    "Emerald",
    "Diamond",
    "Master",
    "Grandmaster",  # Blizzard merges Grandmaster and Champion into this one tier value
]

INPUTS = ["PC", "Console"]

REGIONS = ["Americas", "Europe", "Asia"]

REQUEST_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 10

OUTPUT_PATH = "data/rank_stats.csv"


def fetch_combo(rank, input_value, region):
    params = {
        "role": "Damage",
        "input": input_value,
        "rq": "2",
        "map": "all-maps",
        "region": region,
        "tier": rank,
    }
    response = requests.get(
        RATES_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    payload = response.json()

    selected = payload["rates"]["selected"]

    selected_tier = selected.get("tier")
    if selected_tier != rank:
        raise ValueError(
            f"requested tier '{rank}' but Blizzard echoed back tier '{selected_tier}'"
        )

    selected_input = selected.get("input")
    if selected_input is not None and selected_input != input_value:
        raise ValueError(
            f"requested input '{input_value}' but Blizzard echoed back input '{selected_input}'"
        )

    selected_region = selected.get("region")
    if selected_region is not None and selected_region != region:
        raise ValueError(
            f"requested region '{region}' but Blizzard echoed back region '{selected_region}'"
        )

    return payload


def parse_records(payload, rank, input_value, region):
    records = []
    for row in payload["rates"]["rates"]:
        hero = row.get("hero", {})
        if hero.get("role") != "DAMAGE":
            continue
        cells = row.get("cells", {})
        records.append(
            {
                "hero_id": row.get("id"),
                "hero_name": cells.get("name"),
                "role": hero.get("role"),
                "rank": rank,
                "input": input_value,
                "region": region,
                "winrate": cells.get("winrate"),
                "pickrate": cells.get("pickrate"),
                "banrate": cells.get("banrate"),
            }
        )
    return records


def main():
    all_records = []
    counts_per_combo = {}
    failed_combos = []

    combos = list(itertools.product(RANKS, INPUTS, REGIONS))

    for i, (rank, input_value, region) in enumerate(combos):
        try:
            payload = fetch_combo(rank, input_value, region)
            records = parse_records(payload, rank, input_value, region)
            all_records.extend(records)
            counts_per_combo[(rank, input_value, region)] = len(records)
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Failed to fetch rank='{rank}' input='{input_value}' region='{region}': {e}")
            failed_combos.append((rank, input_value, region))

        if i < len(combos) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(
        all_records,
        columns=[
            "hero_id",
            "hero_name",
            "role",
            "rank",
            "input",
            "region",
            "winrate",
            "pickrate",
            "banrate",
        ],
    )
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDamage heroes returned per rank/input/region combination:")
    for rank, input_value, region in combos:
        count = counts_per_combo.get((rank, input_value, region))
        status = count if count is not None else "FAILED"
        print(f"  rank={rank:<12} input={input_value:<8} region={region:<9} heroes={status}")

    successful_counts = list(counts_per_combo.values())
    if successful_counts:
        most_common_count, _ = Counter(successful_counts).most_common(1)[0]
    else:
        most_common_count = 0

    expected_total = most_common_count * len(counts_per_combo)
    actual_total = len(df)

    dupe_combo_sizes = df.groupby(["hero_id", "rank", "input", "region"]).size()
    duplicate_count = int((dupe_combo_sizes > 1).sum())

    outliers = {
        combo: count
        for combo, count in counts_per_combo.items()
        if count != most_common_count
    }

    print(f"\nMost common hero count per combination: {most_common_count}")
    print(f"Expected total rows (most common count x successful combinations): {expected_total}")
    print(f"Actual total rows: {actual_total}")
    print(f"Duplicate hero_id+rank+input+region combinations: {duplicate_count}")

    if failed_combos:
        print(f"\nFailed combinations ({len(failed_combos)}):")
        for rank, input_value, region in failed_combos:
            print(f"  rank={rank} input={input_value} region={region}")
    else:
        print("\nFailed combinations: none")

    if outliers:
        print(f"\nCombinations with a different hero count ({len(outliers)}):")
        for (rank, input_value, region), count in outliers.items():
            print(f"  rank={rank} input={input_value} region={region} heroes={count}")
    else:
        print("\nCombinations with a different hero count: none")

    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nFirst few rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()
