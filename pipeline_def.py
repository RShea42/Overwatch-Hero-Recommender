"""Custom sklearn transformer implementing the finalized (Formula C) hero
recommendation logic.

This module is self-contained for deployment: the matchup-matrix/pool/
scoring helper functions below are relocated verbatim from
explore_recommendations.py (which remains in the repo as the research/CLI
script) so the Modal image only needs to ship serve.py, pipeline_def.py, and
pipeline.joblib - no runtime import of explore_recommendations.py.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


class HeroRecommenderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, matchups_df=None, rank_df=None):
        self.matchups_df = matchups_df
        self.rank_df = rank_df

    def fit(self, X, y=None):
        # Genuine fitted state: derived once here (not on every transform()
        # call) and stored with the sklearn trailing-underscore convention.
        # These attributes are what sklearn's default check_is_fitted looks
        # for, and what actually gets serialized as "learned" output.
        all_heroes = sorted(
            set(self.matchups_df["hero_id"]) | set(self.matchups_df["opponent_id"])
        )
        self.all_heroes_ = all_heroes
        self.opponent_pool_ = list(all_heroes)
        # Role-specific eligible candidate pools (e.g. "DAMAGE"/"TANK"/
        # "SUPPORT"), derived from rank_df. Matchup evaluation itself stays
        # role-agnostic - opponent_pool_/matchup_dict_ always cover the
        # complete roster regardless of which role's pool is being built.
        self.heroes_by_role_ = {
            role: sorted(group["hero_id"].unique())
            for role, group in self.rank_df.groupby("role")
        }
        self.matchup_dict_ = build_symmetric_matchup_dict(self.matchups_df, all_heroes)
        return self

    def transform(self, X):
        return [self._recommend_for_request(record) for record in X]

    def _get_contextual_stat(self, candidate, column, rank, input_value, region):
        rows = self.rank_df[
            (self.rank_df["hero_id"] == candidate)
            & (self.rank_df["rank"] == rank)
            & (self.rank_df["input"] == input_value)
            & (self.rank_df["region"] == region)
        ]
        if rows.empty:
            return None
        return rows.iloc[0][column]

    def _score_candidates(self, candidate_pool, vulnerabilities, rank, input_value, region):
        """Final deterministic hierarchy: vulnerabilities_resolved desc,
        total_improvement desc, contextual_winrate desc (for the requested
        rank/input/region), candidate ascending. The scoring itself
        (compute_candidate_score, i.e. Formula C) is unchanged; it now reads
        the fitted self.matchup_dict_ instead of a recomputed local dict."""
        rows = []
        for candidate in candidate_pool:
            resolved_count, total_improvement = compute_candidate_score(
                candidate, self.matchup_dict_, vulnerabilities
            )
            rows.append(
                {
                    "candidate": candidate,
                    "vulnerabilities_resolved": resolved_count,
                    "total_improvement": total_improvement,
                    "contextual_winrate": self._get_contextual_stat(
                        candidate, "winrate", rank, input_value, region
                    ),
                    "contextual_pickrate": self._get_contextual_stat(
                        candidate, "pickrate", rank, input_value, region
                    ),
                    "contextual_banrate": self._get_contextual_stat(
                        candidate, "banrate", rank, input_value, region
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(
            ["vulnerabilities_resolved", "total_improvement", "contextual_winrate", "candidate"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    def _recommend_next(self, roster, role, rank, input_value, region):
        """Mirrors the original recommend_next_hero, now reading fitted state
        (self.matchup_dict_, self.heroes_by_role_, self.opponent_pool_)
        instead of rebuilding it. `roster` is an arbitrary list of heroes,
        not hardcoded slots - reusable for evaluating any N-hero pool later.
        Matchup evaluation (pool_values/vulnerabilities) is always computed
        against the complete opponent_pool_, regardless of role; only the
        candidate pool being scored is role-restricted."""
        pool_values = compute_pool_values(self.matchup_dict_, roster, self.opponent_pool_)
        vulnerabilities = get_vulnerabilities(pool_values)

        eligible_heroes = self.heroes_by_role_.get(role, [])
        candidate_pool = [hero for hero in eligible_heroes if hero not in roster]
        scores_df = self._score_candidates(candidate_pool, vulnerabilities, rank, input_value, region)

        return scores_df, vulnerabilities, pool_values

    def _recommend_for_request(self, record):
        heroes = list(record["heroes"])
        role = record["role"]
        rank = record["rank"]
        input_value = record["input"]
        region = record["region"]

        eligible_heroes = self.heroes_by_role_.get(role)
        if not eligible_heroes:
            raise ValueError(f"Unknown or empty role: {role!r}")

        if len(heroes) not in (1, 2):
            raise ValueError(
                f"'heroes' must contain 1 or 2 {role} hero IDs, got {len(heroes)}: {heroes}"
            )
        for hero in heroes:
            if hero not in eligible_heroes:
                raise ValueError(f"Unknown {role} hero id: {hero!r}")

        roster = list(heroes)
        num_recommendations = 2 if len(heroes) == 1 else 1

        recommendations = []
        for _ in range(num_recommendations):
            roster_before = list(roster)

            scores_df, vulnerabilities, _ = self._recommend_next(roster, role, rank, input_value, region)
            winner_row = scores_df.iloc[0]
            recommended_hero = winner_row["candidate"]

            # Explainability lists: the same opponent IDs the existing counts
            # (vulnerabilities_before / vulnerabilities_resolved) are derived
            # from. Resolution check mirrors compute_candidate_score exactly
            # (candidate_value >= 0, skipping opponents with no known value)
            # so vulnerability_heroes_resolved always agrees with the count
            # Formula C itself already produced.
            vulnerability_heroes_before = sorted(vulnerabilities.keys())
            vulnerability_heroes_resolved = sorted(
                opponent
                for opponent in vulnerabilities
                if get_matchup_value(self.matchup_dict_, recommended_hero, opponent) is not None
                and get_matchup_value(self.matchup_dict_, recommended_hero, opponent) >= 0
            )

            roster.append(recommended_hero)
            pool_values_after = compute_pool_values(self.matchup_dict_, roster, self.opponent_pool_)
            vulnerabilities_after = get_vulnerabilities(pool_values_after)
            remaining_negative_count = len(vulnerabilities_after)
            remaining_negative_heroes = sorted(vulnerabilities_after.keys())

            recommendations.append(
                {
                    "roster_before": roster_before,
                    "recommended_hero": recommended_hero,
                    "vulnerabilities_before": len(vulnerabilities),
                    "vulnerabilities_resolved": int(winner_row["vulnerabilities_resolved"]),
                    "remaining_negative_count": remaining_negative_count,
                    "vulnerability_heroes_before": vulnerability_heroes_before,
                    "vulnerability_heroes_resolved": vulnerability_heroes_resolved,
                    "remaining_negative_heroes": remaining_negative_heroes,
                    "contextual_winrate": winner_row["contextual_winrate"],
                    "contextual_pickrate": winner_row["contextual_pickrate"],
                    "contextual_banrate": winner_row["contextual_banrate"],
                }
            )

        return {
            "input_heroes": heroes,
            "role": role,
            "rank": rank,
            "input": input_value,
            "region": region,
            "final_roster": roster,
            "recommendations": recommendations,
        }
