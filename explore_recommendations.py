import pandas as pd

MATCHUPS_PATH = "data/matchups.csv"
RANK_STATS_PATH = "data/rank_stats.csv"

RANK = "Silver"
INPUT_VALUE = "PC"
REGION = "Americas"


def load_data():
    matchups_df = pd.read_csv(MATCHUPS_PATH)
    rank_df = pd.read_csv(RANK_STATS_PATH)
    return matchups_df, rank_df


def build_directional_lookup(matchups_df):
    lookup = {}
    for hero, opponent, value in matchups_df[
        ["hero_id", "opponent_id", "resolved_value"]
    ].itertuples(index=False):
        lookup[(hero, opponent)] = value
    return lookup


def build_symmetric_matchup_dict(matchups_df, all_heroes):
    """Derive a symmetric matrix from the raw directional resolved_value.

    symmetric(A,B) = (resolved(A->B) - resolved(B->A)) / 2, defined only when
    BOTH directional records exist; symmetric(B,A) is set to -symmetric(A,B) by
    construction. Self-matchups are exactly 0. A pair missing either direction
    is left out entirely rather than imputed.
    """
    raw = build_directional_lookup(matchups_df)
    matchup_dict = {hero: {hero: 0.0} for hero in all_heroes}

    for i, hero_a in enumerate(all_heroes):
        for hero_b in all_heroes[i + 1 :]:
            forward = raw.get((hero_a, hero_b))
            reverse = raw.get((hero_b, hero_a))
            if forward is None or reverse is None:
                continue
            symmetric_value = (forward - reverse) / 2
            matchup_dict[hero_a][hero_b] = symmetric_value
            matchup_dict[hero_b][hero_a] = -symmetric_value

    return matchup_dict


def get_matchup_value(matchup_dict, hero, opponent):
    return matchup_dict.get(hero, {}).get(opponent)


def compute_pool_values(matchup_dict, roster, opponent_pool):
    """Best (max) resolved matchup value available among roster heroes, per opponent."""
    pool_values = {}
    for opponent in opponent_pool:
        values = [
            v
            for v in (get_matchup_value(matchup_dict, hero, opponent) for hero in roster)
            if v is not None
        ]
        if values:
            pool_values[opponent] = max(values)
    return pool_values


def get_vulnerabilities(pool_values):
    return {opponent: value for opponent, value in pool_values.items() if value < 0}


def compute_candidate_score(candidate, matchup_dict, vulnerabilities):
    """Formula C: count of current known negative matchups the candidate would
    resolve to >= 0, with total (unweighted) improvement as the tie-break value.

    Only opponents already present in `vulnerabilities` (known, current pool
    value < 0) are considered. A candidate with no known value against an
    opponent contributes nothing for that opponent - it is never treated as 0
    and never earns credit merely for having *any* known value.
    """
    resolved_count = 0
    total_improvement = 0.0

    for opponent, current_value in vulnerabilities.items():
        candidate_value = get_matchup_value(matchup_dict, candidate, opponent)
        if candidate_value is None:
            continue

        if candidate_value >= 0:
            resolved_count += 1

        total_improvement += max(candidate_value - current_value, 0)

    return resolved_count, total_improvement


def get_contextual_stat(rank_df, candidate, column):
    rows = rank_df[
        (rank_df["hero_id"] == candidate)
        & (rank_df["rank"] == RANK)
        & (rank_df["input"] == INPUT_VALUE)
        & (rank_df["region"] == REGION)
    ]
    if rows.empty:
        return None
    return rows.iloc[0][column]


def score_candidates(candidate_pool, matchup_dict, vulnerabilities, rank_df):
    """Rank candidates by the final deterministic hierarchy:

    1. vulnerabilities_resolved (descending) - absolute primary criterion.
    2. total_improvement (descending) - matchup-quality tie-break among
       candidates resolving the same number of vulnerabilities.
    3. contextual_winrate (descending), for the fixed RANK/INPUT_VALUE/REGION
       context - only ever consulted once the first two criteria are tied,
       so it can never let a candidate with fewer vulnerabilities resolved or
       lower total_improvement outrank another candidate.
    4. candidate name (ascending) - final deterministic fallback.

    contextual_pickrate and contextual_banrate are computed and included for
    display only; neither participates in the sort.
    """
    rows = []
    for candidate in candidate_pool:
        resolved_count, total_improvement = compute_candidate_score(
            candidate, matchup_dict, vulnerabilities
        )
        rows.append(
            {
                "candidate": candidate,
                "vulnerabilities_resolved": resolved_count,
                "total_improvement": total_improvement,
                "contextual_winrate": get_contextual_stat(rank_df, candidate, "winrate"),
                "contextual_pickrate": get_contextual_stat(rank_df, candidate, "pickrate"),
                "contextual_banrate": get_contextual_stat(rank_df, candidate, "banrate"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["vulnerabilities_resolved", "total_improvement", "contextual_winrate", "candidate"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def recommend_next_hero(roster, damage_heroes, matchup_dict, opponent_pool, rank_df):
    """Score every Damage hero not already in `roster` as the next addition to
    it, under Formula C. `roster` is an arbitrary list of heroes (any size),
    not a hardcoded "main hero"/"secondary hero" pair - this is reusable for
    evaluating any N-hero pool. Returns (scores_df, vulnerabilities, pool_values)
    for the CURRENT roster (i.e. before adding a recommendation)."""
    pool_values = compute_pool_values(matchup_dict, roster, opponent_pool)
    vulnerabilities = get_vulnerabilities(pool_values)

    candidate_pool = [hero for hero in damage_heroes if hero not in roster]
    scores_df = score_candidates(candidate_pool, matchup_dict, vulnerabilities, rank_df)

    return scores_df, vulnerabilities, pool_values


def run_experiment_for_hero(starting_hero, matchup_dict, opponent_pool, damage_heroes, rank_df):
    """Run the two-slot sequential process for one starting hero using
    `recommend_next_hero`. Returns the summary row plus the full slot #2/#3
    candidate tables (for the top-5 display) without printing anything."""
    roster = [starting_hero]

    slot2_scores, starting_vulnerabilities, pool_values = recommend_next_hero(
        roster, damage_heroes, matchup_dict, opponent_pool, rank_df
    )
    slot2_winner = slot2_scores.iloc[0]["candidate"]
    roster.append(slot2_winner)

    slot3_scores, vulnerabilities_2, _ = recommend_next_hero(
        roster, damage_heroes, matchup_dict, opponent_pool, rank_df
    )
    slot3_winner = slot3_scores.iloc[0]["candidate"]
    roster.append(slot3_winner)

    pool_values_3 = compute_pool_values(matchup_dict, roster, opponent_pool)
    final_negative = len(get_vulnerabilities(pool_values_3))

    summary = {
        "starting_hero": starting_hero,
        "starting_known": len(pool_values),
        "starting_negative": len(starting_vulnerabilities),
        "slot_2_recommendation": slot2_winner,
        "after_slot_2_negative": len(vulnerabilities_2),
        "slot_3_recommendation": slot3_winner,
        "final_negative": final_negative,
        "slot3_eliminated": len(vulnerabilities_2) - final_negative,
    }

    return summary, slot2_scores, slot3_scores


def print_top5(label, scores_df):
    top5 = scores_df.head(5)[
        ["candidate", "vulnerabilities_resolved", "total_improvement", "contextual_winrate"]
    ]
    print(f"\n{label}:")
    print(top5.to_string(index=False))


def main():
    matchups_df, rank_df = load_data()

    all_heroes = sorted(set(matchups_df["hero_id"]) | set(matchups_df["opponent_id"]))
    opponent_pool = list(all_heroes)  # fixed universe of 53, unchanged across stages
    damage_heroes = sorted(rank_df["hero_id"].unique())

    matchup_dict = build_symmetric_matchup_dict(matchups_df, all_heroes)

    summaries = []
    top5_tables = {}
    selected_slots = []

    for starting_hero in damage_heroes:
        summary, slot2_scores, slot3_scores = run_experiment_for_hero(
            starting_hero, matchup_dict, opponent_pool, damage_heroes, rank_df
        )
        summaries.append(summary)
        top5_tables[starting_hero] = (slot2_scores, slot3_scores)
        selected_slots.append(summary["slot_2_recommendation"])
        selected_slots.append(summary["slot_3_recommendation"])

    summary_df = pd.DataFrame(summaries)
    print("Summary across all 24 starting Damage heroes (Formula C):")
    print(
        summary_df[
            [
                "starting_hero",
                "starting_negative",
                "slot_2_recommendation",
                "after_slot_2_negative",
                "slot_3_recommendation",
                "final_negative",
            ]
        ].to_string(index=False)
    )

    print("\nFinal negative matchup count distribution:")
    print(f"  mean:   {summary_df['final_negative'].mean():.4f}")
    print(f"  median: {summary_df['final_negative'].median():.4f}")
    print(f"  min:    {summary_df['final_negative'].min()}")
    print(f"  max:    {summary_df['final_negative'].max()}")

    zero_gain = summary_df[summary_df["slot3_eliminated"] == 0]
    print(f"\nStarting heroes where slot #3 eliminates zero additional vulnerabilities: {len(zero_gain)}")
    if not zero_gain.empty:
        print(zero_gain[["starting_hero", "slot_2_recommendation", "slot_3_recommendation"]].to_string(index=False))

    for starting_hero in ("soldier-76", "tracer", "reaper"):
        slot2_scores, slot3_scores = top5_tables[starting_hero]
        print(f"\n=== {starting_hero} ===")
        print_top5(f"Top 5 slot #2 candidates for {starting_hero}", slot2_scores)
        print_top5(f"Top 5 slot #3 candidates for {starting_hero}", slot3_scores)

    distinct_heroes = sorted(set(selected_slots))
    print(f"\nDistinct heroes across all {len(selected_slots)} selected recommendation slots: {len(distinct_heroes)}")
    print(distinct_heroes)


if __name__ == "__main__":
    main()
