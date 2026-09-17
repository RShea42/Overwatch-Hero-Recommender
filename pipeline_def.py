"""Custom sklearn transformer wrapping the finalized (Formula C) hero
recommendation logic from explore_recommendations.py. The recommendation
algorithm itself is imported and reused unchanged; this module only adds a
runtime-parameterized version of the contextual (rank/input/region) lookup
and candidate sort, since the original module hardcodes a single fixed
context (RANK/INPUT_VALUE/REGION module-level constants) and this pipeline
must accept rank/input/region as per-request input.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import explore_recommendations as er


class HeroRecommenderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, matchups_df=None, rank_df=None):
        self.matchups_df = matchups_df
        self.rank_df = rank_df

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        # This transformer holds no learned state (it wraps a fixed
        # deterministic formula over already-loaded data), so there is no
        # fitted attribute for sklearn's default check_is_fitted to detect.
        # This hook is the standard way to declare "always fitted" without
        # adding anything to fit() beyond `return self`.
        return True

    def transform(self, X):
        all_heroes = sorted(
            set(self.matchups_df["hero_id"]) | set(self.matchups_df["opponent_id"])
        )
        opponent_pool = list(all_heroes)
        damage_heroes = sorted(self.rank_df["hero_id"].unique())
        matchup_dict = er.build_symmetric_matchup_dict(self.matchups_df, all_heroes)

        return [
            self._recommend_for_request(
                record, damage_heroes, opponent_pool, matchup_dict
            )
            for record in X
        ]

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

    def _score_candidates(self, candidate_pool, matchup_dict, vulnerabilities, rank, input_value, region):
        """Same deterministic hierarchy as explore_recommendations.score_candidates:
        vulnerabilities_resolved desc, total_improvement desc, contextual_winrate
        desc (for the requested rank/input/region), candidate ascending. The
        scoring itself (er.compute_candidate_score, i.e. Formula C) is reused
        unchanged; only the context used for the winrate tiebreak is a runtime
        parameter here instead of a hardcoded module constant."""
        rows = []
        for candidate in candidate_pool:
            resolved_count, total_improvement = er.compute_candidate_score(
                candidate, matchup_dict, vulnerabilities
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

    def _recommend_next(self, roster, damage_heroes, matchup_dict, opponent_pool, rank, input_value, region):
        """Mirrors explore_recommendations.recommend_next_hero, generalized to
        accept rank/input/region as parameters. `roster` is an arbitrary list
        of heroes, not hardcoded slots - reusable for evaluating any N-hero
        pool later."""
        pool_values = er.compute_pool_values(matchup_dict, roster, opponent_pool)
        vulnerabilities = er.get_vulnerabilities(pool_values)

        candidate_pool = [hero for hero in damage_heroes if hero not in roster]
        scores_df = self._score_candidates(
            candidate_pool, matchup_dict, vulnerabilities, rank, input_value, region
        )

        return scores_df, vulnerabilities, pool_values

    def _recommend_for_request(self, record, damage_heroes, opponent_pool, matchup_dict):
        heroes = list(record["heroes"])
        rank = record["rank"]
        input_value = record["input"]
        region = record["region"]

        if len(heroes) not in (1, 2):
            raise ValueError(
                f"'heroes' must contain 1 or 2 Damage hero IDs, got {len(heroes)}: {heroes}"
            )
        for hero in heroes:
            if hero not in damage_heroes:
                raise ValueError(f"Unknown Damage hero id: {hero!r}")

        roster = list(heroes)
        num_recommendations = 2 if len(heroes) == 1 else 1

        recommendations = []
        for _ in range(num_recommendations):
            roster_before = list(roster)

            scores_df, vulnerabilities, _ = self._recommend_next(
                roster, damage_heroes, matchup_dict, opponent_pool, rank, input_value, region
            )
            winner_row = scores_df.iloc[0]
            recommended_hero = winner_row["candidate"]

            roster.append(recommended_hero)
            pool_values_after = er.compute_pool_values(matchup_dict, roster, opponent_pool)
            remaining_negative_count = len(er.get_vulnerabilities(pool_values_after))

            recommendations.append(
                {
                    "roster_before": roster_before,
                    "recommended_hero": recommended_hero,
                    "vulnerabilities_before": len(vulnerabilities),
                    "vulnerabilities_resolved": int(winner_row["vulnerabilities_resolved"]),
                    "remaining_negative_count": remaining_negative_count,
                    "contextual_winrate": winner_row["contextual_winrate"],
                    "contextual_pickrate": winner_row["contextual_pickrate"],
                    "contextual_banrate": winner_row["contextual_banrate"],
                }
            )

        return {
            "input_heroes": heroes,
            "rank": rank,
            "input": input_value,
            "region": region,
            "final_roster": roster,
            "recommendations": recommendations,
        }
