import itertools
import os
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

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2

OUTPUT_PATH = "data/rank_stats.csv"

CSV_COLUMNS = [
    "hero_id",
    "hero_name",
    "role",
    "rank",
    "input",
    "region",
    "winrate",
    "pickrate",
    "banrate",
]


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


def fetch_combo_with_retry(rank, input_value, region):
    """Bounded retry with a short backoff for transient failures (e.g. the
    read timeouts observed against Blizzard's endpoint for some Europe/Asia
    contexts). Returns the payload on success, or None once MAX_ATTEMPTS is
    exhausted - never retries indefinitely."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fetch_combo(rank, input_value, region)
        except (requests.exceptions.RequestException, ValueError) as e:
            print(
                f"  attempt {attempt}/{MAX_ATTEMPTS} failed for "
                f"rank={rank} input={input_value} region={region}: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    return None


def parse_records(payload, rank, input_value, region):
    records = []
    for row in payload["rates"]["rates"]:
        hero = row.get("hero", {})
        if hero.get("role") not in ("DAMAGE", "TANK", "SUPPORT"):
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


def load_existing(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[CSV_COLUMNS]


def determine_reference_hero_set(existing_df):
    """The expected hero roster for a *complete* context, derived from the
    data itself (the largest hero_id set among existing contexts) rather
    than a hardcoded hero count - avoids a fragile "53" assumption while
    still giving a real, checkable definition of "complete"."""
    if existing_df.empty:
        return frozenset()

    combo_hero_sets = existing_df.groupby(["rank", "input", "region"])["hero_id"].apply(
        lambda s: frozenset(s)
    )
    if combo_hero_sets.empty:
        return frozenset()

    # Prefer the most common hero set among existing contexts (mode), so a
    # single corrupted/partial legacy context can't set a bad reference.
    counts = Counter(combo_hero_sets.values)
    reference_set, _ = counts.most_common(1)[0]
    return reference_set


def find_complete_combos(existing_df, reference_hero_set):
    if not reference_hero_set:
        return set()

    complete = set()
    for combo, group in existing_df.groupby(["rank", "input", "region"]):
        if frozenset(group["hero_id"]) == reference_hero_set:
            complete.add(combo)
    return complete


def main():
    all_combos = list(itertools.product(RANKS, INPUTS, REGIONS))

    existing_df = load_existing(OUTPUT_PATH)
    reference_hero_set = determine_reference_hero_set(existing_df)
    complete_combos = find_complete_combos(existing_df, reference_hero_set)

    combos_to_fetch = [c for c in all_combos if c not in complete_combos]

    print(f"Existing rows loaded: {len(existing_df)}")
    print(f"Reference hero roster size (from existing data): {len(reference_hero_set)}")
    print(f"Already-complete contexts (skipped): {len(complete_combos)} of {len(all_combos)}")
    print(f"Contexts to fetch this run: {len(combos_to_fetch)}")

    newly_fetched_records = []
    newly_fetched_combos = set()
    failed_combos = []
    total_attempts_used = 0

    for i, (rank, input_value, region) in enumerate(combos_to_fetch):
        print(f"\nFetching rank={rank} input={input_value} region={region} ...")
        payload = fetch_combo_with_retry(rank, input_value, region)

        if payload is None:
            print(f"  giving up after {MAX_ATTEMPTS} attempts")
            failed_combos.append((rank, input_value, region))
        else:
            records = parse_records(payload, rank, input_value, region)
            newly_fetched_records.extend(records)
            newly_fetched_combos.add((rank, input_value, region))
            print(f"  success: {len(records)} hero rows")

            if not reference_hero_set:
                reference_hero_set = frozenset(r["hero_id"] for r in records)

        if i < len(combos_to_fetch) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    # Merge-safe write: keep every existing row EXCEPT for contexts we
    # successfully refetched this run (those get atomically replaced by the
    # fresh response). Contexts that failed again this run keep whatever was
    # already on disk for them (nothing is deleted on failure).
    if newly_fetched_combos:
        combo_tuples = existing_df[["rank", "input", "region"]].apply(tuple, axis=1)
        kept_existing = existing_df[~combo_tuples.isin(newly_fetched_combos)]
    else:
        kept_existing = existing_df

    new_df = pd.DataFrame(newly_fetched_records, columns=CSV_COLUMNS)
    final_df = pd.concat([kept_existing, new_df], ignore_index=True)
    final_df = final_df[CSV_COLUMNS]
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total rows: {len(final_df)}")
    if failed_combos:
        print(f"\nContexts still failing after retries ({len(failed_combos)}):")
        for rank, input_value, region in failed_combos:
            print(f"  rank={rank} input={input_value} region={region}")
    else:
        print("\nAll requested contexts fetched successfully (or were already complete).")

    validate_dataset(final_df, all_combos, reference_hero_set)


def validate_dataset(df, all_combos, reference_hero_set):
    print("\n" + "=" * 70)
    print("Post-collection validation")
    print("=" * 70)

    present_combos = set(df[["rank", "input", "region"]].apply(tuple, axis=1))
    missing_combos = [c for c in all_combos if c not in present_combos]
    extra_combos = [c for c in present_combos if c not in all_combos]

    print(f"Expected contexts: {len(all_combos)}")
    print(f"Present contexts: {len(present_combos)}")
    if missing_combos:
        print(f"MISSING contexts ({len(missing_combos)}):")
        for c in missing_combos:
            print(f"  {c}")
    else:
        print("All 48 expected contexts present: YES")

    if extra_combos:
        print(f"UNEXPECTED extra contexts ({len(extra_combos)}):")
        for c in extra_combos:
            print(f"  {c}")
    else:
        print("No unexpected extra contexts: YES")

    wrong_roster = []
    missing_role = []
    for combo, group in df.groupby(["rank", "input", "region"]):
        hero_set = frozenset(group["hero_id"])
        if reference_hero_set and hero_set != reference_hero_set:
            wrong_roster.append((combo, len(hero_set)))
        roles_present = set(group["role"].unique())
        if not {"DAMAGE", "TANK", "SUPPORT"}.issubset(roles_present):
            missing_role.append((combo, roles_present))

    if wrong_roster:
        print(f"Contexts with unexpected hero roster ({len(wrong_roster)}):")
        for combo, count in wrong_roster:
            print(f"  {combo}: {count} heroes (expected {len(reference_hero_set)})")
    else:
        print(f"Every present context has the expected {len(reference_hero_set)}-hero roster: YES")

    if missing_role:
        print(f"Contexts missing a role ({len(missing_role)}):")
        for combo, roles in missing_role:
            print(f"  {combo}: roles present = {roles}")
    else:
        print("All three roles (DAMAGE/TANK/SUPPORT) represented in every context: YES")

    dupes = df.duplicated(subset=["hero_id", "rank", "input", "region"]).sum()
    print(f"Duplicate (hero_id, rank, input, region) rows: {dupes}")

    all_ok = (
        not missing_combos
        and not extra_combos
        and not wrong_roster
        and not missing_role
        and dupes == 0
    )
    print(f"\nDataset fully complete and valid (48/48): {all_ok}")
    return all_ok


if __name__ == "__main__":
    main()
