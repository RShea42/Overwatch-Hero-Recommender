"""Custom sklearn transformer implementing the finalized (Formula C) hero
recommendation logic.

This module is self-contained for deployment: the matchup-matrix/pool/
scoring helper functions below are relocated verbatim from
explore_recommendations.py (which remains in the repo as the research/CLI
script) so the Modal image only needs to ship serve.py, pipeline_def.py, and
pipeline.joblib - no runtime import of explore_recommendations.py.
"""

import itertools

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


def evaluate_complete_pool(matchup_dict, main, backups, opponent_pool, original_vulnerabilities=None):
    """Shared complete-pool evaluator for {main} + backups (currently 2
    backups for Builder/Completer; the signature already generalizes to any
    number of backups so a future no-search "Evaluator" mode can call this
    directly without new matchup-scoring logic).

    Reuses compute_pool_values/get_vulnerabilities completely unchanged -
    the same max-known-value pool semantics and unknown-data handling used
    everywhere else in this module. An unknown candidate matchup value is
    never substituted with 0/neutral: compute_pool_values only ever takes
    the max of KNOWN values, so an opponent no one in the roster has data
    for simply never enters pool_values, and an opponent only `main` has
    data for keeps main's own (possibly negative) value untouched.

    `original_vulnerabilities` (the main-alone V1) may be passed in by a
    caller running many candidates against the same main, to avoid
    recomputing it once per candidate; if omitted it is derived here.
    """
    if original_vulnerabilities is None:
        main_pool_values = compute_pool_values(matchup_dict, [main], opponent_pool)
        original_vulnerabilities = get_vulnerabilities(main_pool_values)

    roster = [main] + list(backups)
    pool_values = compute_pool_values(matchup_dict, roster, opponent_pool)
    residual_vulnerabilities = get_vulnerabilities(pool_values)

    resolved_opponents = sorted(
        opp for opp in original_vulnerabilities if opp not in residual_vulnerabilities
    )
    # Always >= 0: pool_values is a max() over a roster that includes `main`,
    # so at any opponent main already had a known value for, the pool value
    # can only stay the same or improve - never get worse.
    total_improvement = sum(
        pool_values[opp] - original_vulnerabilities[opp] for opp in original_vulnerabilities
    )

    if residual_vulnerabilities:
        worst_opponent = min(residual_vulnerabilities, key=residual_vulnerabilities.get)
        worst_value = residual_vulnerabilities[worst_opponent]
    else:
        # Zero residual vulnerabilities: this pool is the safest possible
        # matchup state. No fake opponent is invented here - both fields
        # stay None, and callers/consumers treat that as "nothing remains".
        worst_opponent = None
        worst_value = None

    total_severity = sum(abs(v) for v in residual_vulnerabilities.values())

    # Unknown-data relevant to explaining THIS pool: original vulnerabilities
    # where every backup lacks a known matchup value against that opponent,
    # so main's original negative value simply carries through untouched -
    # not because the backups are known to be bad there, but because there
    # is no data to resolve or worsen it either way.
    unknown_relevant_opponents = {
        opp: [b for b in backups if get_matchup_value(matchup_dict, b, opp) is None]
        for opp in original_vulnerabilities
        if all(get_matchup_value(matchup_dict, b, opp) is None for b in backups)
    }

    return {
        "original_vulnerabilities": original_vulnerabilities,
        "residual_vulnerabilities": residual_vulnerabilities,
        "residual_count": len(residual_vulnerabilities),
        "worst_residual_opponent": worst_opponent,
        "worst_residual_value": worst_value,
        "total_residual_severity": total_severity,
        "total_matchup_improvement": total_improvement,
        "resolved_opponents": resolved_opponents,
        "unknown_relevant_opponents": unknown_relevant_opponents,
    }


def pareto_filter(records, dims):
    """Remove dominated records. `dims` is a list of (key, 'min'|'max').

    A record x is dominated and removed only if some other record y is no
    worse than x on EVERY listed dimension and strictly better on at least
    one - exactly the dominance test used throughout the exhaustive-pair
    diagnostic. No redundancy penalty and no unknown-data bonus/penalty are
    part of this: those stay explanatory metadata computed elsewhere.
    """

    def not_worse_and_strictly_better(y, x):
        at_least_as_good = True
        strictly_better = False
        for key, direction in dims:
            yv, xv = y[key], x[key]
            if direction == "min":
                if yv > xv:
                    at_least_as_good = False
                    break
                if yv < xv:
                    strictly_better = True
            else:
                if yv < xv:
                    at_least_as_good = False
                    break
                if yv > xv:
                    strictly_better = True
        return at_least_as_good and strictly_better

    dominated = set()
    for i, x in enumerate(records):
        for j, y in enumerate(records):
            if i == j:
                continue
            if not_worse_and_strictly_better(y, x):
                dominated.add(i)
                break
    return [r for i, r in enumerate(records) if i not in dominated]


def select_candidate4(records, wr_primary_key, wr_secondary_key, identity_key):
    """STRICT exact-tie lexicographic selection ("Candidate 4"), exactly as
    specified: worst residual matchup -> total residual severity ->
    contextual WR (minimum-of-backups, then mean-of-backups) -> alphabetical
    fallback. No tolerances, no "close enough", no weighted sums, no fitted
    coefficients, no severity bands, no squared-severity score. A later tier
    is consulted only when every earlier tier is EXACTLY tied.

    `wr_primary_key`/`wr_secondary_key` let Builder (min_backup_wr /
    mean_backup_wr) and Completer (both mapped to the candidate Tertiary's
    own WR - see report) reuse this identical selection function.
    `identity_key(record)` must return an already-normalized, hero-id-sorted
    tuple so the alphabetical fallback is deterministic (unordered Builder
    pairs must be pre-sorted by the caller before this key is applied).

    Returns (winner_record, decisive_tier_name).
    """

    def effective_worst(r):
        # A zero-residual pool's worst_residual_value is None; 0.0 is a safe
        # sentinel here because every real residual value is strictly < 0
        # (get_vulnerabilities only keeps negative values), so 0.0 can never
        # collide with a real residual and always sorts as strictly safer.
        return r["worst_residual_value"] if r["worst_residual_value"] is not None else 0.0

    def none_low(v):
        # Missing contextual WR data must sort as worse than any real
        # winrate percentage, never as neutral/average - -inf is the
        # mathematically exact "worse than everything real" value, not a
        # fitted or tuned number.
        return v if v is not None else float("-inf")

    pool = list(records)

    best_worst = max(effective_worst(r) for r in pool)
    tier1 = [r for r in pool if effective_worst(r) == best_worst]
    if len(tier1) == 1:
        return tier1[0], "worst_case_alone"

    best_sev = min(r["total_residual_severity"] for r in tier1)
    tier2 = [r for r in tier1 if r["total_residual_severity"] == best_sev]
    if len(tier2) == 1:
        return tier2[0], "severity_after_worstcase_tie"

    best_wr_primary = max(none_low(r[wr_primary_key]) for r in tier2)
    tier3a = [r for r in tier2 if none_low(r[wr_primary_key]) == best_wr_primary]
    if len(tier3a) == 1:
        return tier3a[0], "contextual_wr_primary_after_matchup_ties"

    best_wr_secondary = max(none_low(r[wr_secondary_key]) for r in tier3a)
    tier3b = [r for r in tier3a if none_low(r[wr_secondary_key]) == best_wr_secondary]
    if len(tier3b) == 1:
        return tier3b[0], "contextual_wr_secondary_after_matchup_and_primary_wr_ties"

    tier4 = sorted(tier3b, key=identity_key)
    return tier4[0], "alphabetical_fallback"


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
        # NEW complete-pool architecture (Builder/Completer). The old
        # sequential Hero#2-then-Hero#3 greedy path
        # (_recommend_next/_recommend_for_request below) is intentionally
        # left completely unmodified and still directly callable - that is
        # how old-vs-new behavior stays comparable without a second live
        # deployment.
        return [self.recommend_pool(record) for record in X]

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

    # ------------------------------------------------------------------
    # NEW complete-pool architecture (Builder / Completer). Everything above
    # this point is the original sequential Formula-C path, left unchanged.
    # ------------------------------------------------------------------

    def _get_wr_index(self):
        # Built lazily from self.rank_df (itself already part of the fitted
        # state) on first use, not during fit(). This is a plain performance
        # cache over already-loaded data - not a re-derivation of the
        # learned matchup state from scratch - so it does not affect the
        # "genuinely fitted pipeline" requirement.
        if not hasattr(self, "_wr_index_"):
            self._wr_index_ = (
                self.rank_df.set_index(["hero_id", "role", "rank", "input", "region"])["winrate"]
                .to_dict()
            )
        return self._wr_index_

    def _contextual_wr(self, hero, role, rank, input_value, region):
        return self._get_wr_index().get((hero, role, rank, input_value, region))

    def _builder_candidate_record(self, main, a, b, role, rank, input_value, region, original_vulnerabilities):
        evaluation = evaluate_complete_pool(
            self.matchup_dict_, main, [a, b], self.opponent_pool_, original_vulnerabilities
        )
        wr_a = self._contextual_wr(a, role, rank, input_value, region)
        wr_b = self._contextual_wr(b, role, rank, input_value, region)
        known_wrs = [w for w in (wr_a, wr_b) if w is not None]
        min_wr = min(known_wrs) if known_wrs else None
        mean_wr = (sum(known_wrs) / len(known_wrs)) if known_wrs else None

        record = dict(evaluation)
        record.update(
            {
                "backup_a": a,
                "backup_b": b,
                "wr_a": wr_a,
                "wr_b": wr_b,
                "min_backup_wr": min_wr,
                "mean_backup_wr": mean_wr,
            }
        )
        return record

    def _completer_candidate_record(self, main, secondary, tertiary, role, rank, input_value, region, original_vulnerabilities):
        evaluation = evaluate_complete_pool(
            self.matchup_dict_, main, [secondary, tertiary], self.opponent_pool_, original_vulnerabilities
        )
        wr_secondary = self._contextual_wr(secondary, role, rank, input_value, region)
        wr_tertiary = self._contextual_wr(tertiary, role, rank, input_value, region)

        record = dict(evaluation)
        record.update(
            {
                "secondary": secondary,
                "tertiary": tertiary,
                "wr_secondary": wr_secondary,
                "wr_tertiary": wr_tertiary,
                # Completer WR objective (both Pareto and Candidate-4 tier 3):
                # the candidate Tertiary's OWN contextual WR, used directly -
                # not folded into a min()/mean() with the Secondary's WR.
                # The Secondary is fixed by the user for every candidate in
                # this search, so it is a constant, not a decision variable;
                # using min(secondary_wr, tertiary_wr) would silently
                # collapse discrimination among every candidate whose own WR
                # exceeds that constant (they would all "tie" at
                # secondary_wr). The Secondary's WR is still returned as
                # explanatory metadata (wr_secondary above / the response's
                # contextual_wr.secondary), just not used as a selection
                # variable. See the implementation report for the full
                # reasoning - this was flagged as a required design decision
                # rather than changed silently.
                "min_backup_wr": wr_tertiary,
                "mean_backup_wr": wr_tertiary,
            }
        )
        return record

    _PARETO_DIMS = [
        ("residual_count", "min"),
        ("total_residual_severity", "min"),
        ("total_matchup_improvement", "max"),
        ("min_backup_wr", "max"),
        ("mean_backup_wr", "max"),
    ]

    def recommend_builder(self, main, role, rank, input_value, region):
        """Mode A: Main only supplied. Evaluate every unordered pair of
        eligible same-role backups as a COMPLETE pool {main, A, B} - not a
        greedy Hero#2-then-Hero#3 search."""
        eligible = self.heroes_by_role_.get(role, [])
        candidates = [h for h in eligible if h != main]

        main_pool_values = compute_pool_values(self.matchup_dict_, [main], self.opponent_pool_)
        original_vulnerabilities = get_vulnerabilities(main_pool_values)

        records = [
            self._builder_candidate_record(main, a, b, role, rank, input_value, region, original_vulnerabilities)
            for a, b in itertools.combinations(candidates, 2)
        ]

        frontier = pareto_filter(records, self._PARETO_DIMS)
        winner, tier = select_candidate4(
            frontier,
            wr_primary_key="min_backup_wr",
            wr_secondary_key="mean_backup_wr",
            identity_key=lambda r: tuple(sorted([r["backup_a"], r["backup_b"]])),
        )

        return {
            "mode": "builder",
            "main": main,
            "secondary": None,
            "candidate_pool_size": len(records),
            "pareto_frontier_size": len(frontier),
            "decisive_tier": tier,
            "winner": winner,
            "original_vulnerabilities": original_vulnerabilities,
        }

    def recommend_completer(self, main, secondary, role, rank, input_value, region):
        """Mode B: Main + Secondary supplied, both fixed/non-negotiable.
        Evaluate every eligible same-role Tertiary as a COMPLETE pool
        {main, secondary, X} using the same evaluator/Pareto/Candidate-4
        pipeline as Builder - this is not the old greedy "Hero #3" step."""
        eligible = self.heroes_by_role_.get(role, [])
        candidates = [h for h in eligible if h not in (main, secondary)]

        main_pool_values = compute_pool_values(self.matchup_dict_, [main], self.opponent_pool_)
        original_vulnerabilities = get_vulnerabilities(main_pool_values)

        records = [
            self._completer_candidate_record(main, secondary, t, role, rank, input_value, region, original_vulnerabilities)
            for t in candidates
        ]

        frontier = pareto_filter(records, self._PARETO_DIMS)
        winner, tier = select_candidate4(
            frontier,
            wr_primary_key="min_backup_wr",
            wr_secondary_key="mean_backup_wr",
            identity_key=lambda r: (r["tertiary"],),
        )

        return {
            "mode": "completer",
            "main": main,
            "secondary": secondary,
            "candidate_pool_size": len(records),
            "pareto_frontier_size": len(frontier),
            "decisive_tier": tier,
            "winner": winner,
            "original_vulnerabilities": original_vulnerabilities,
        }

    def recommend_pool(self, record):
        """Dispatcher: secondary omitted/None -> Builder, secondary supplied
        -> Completer. Mirrors the defensive validation style already used by
        _recommend_for_request (role/membership checks raise ValueError;
        serve.py's Pydantic layer is expected to catch most of this first,
        but the pipeline itself never trusts unvalidated input)."""
        role = record["role"]
        main = record["main"]
        secondary = record.get("secondary")
        rank = record["rank"]
        input_value = record["input"]
        region = record["region"]

        eligible_heroes = self.heroes_by_role_.get(role)
        if not eligible_heroes:
            raise ValueError(f"Unknown or empty role: {role!r}")
        if main not in eligible_heroes:
            raise ValueError(f"Unknown {role} hero id for main: {main!r}")
        if secondary is not None:
            if secondary not in eligible_heroes:
                raise ValueError(f"Unknown {role} hero id for secondary: {secondary!r}")
            if secondary == main:
                raise ValueError("main and secondary must be distinct heroes")

        if secondary is None:
            result = self.recommend_builder(main, role, rank, input_value, region)
        else:
            result = self.recommend_completer(main, secondary, role, rank, input_value, region)

        return self._format_pool_response(result, role, rank, input_value, region)

    def _format_pool_response(self, result, role, rank, input_value, region):
        mode = result["mode"]
        winner = result["winner"]
        original_vulnerabilities = result["original_vulnerabilities"]

        original_list = [
            {"opponent": opp, "value": val}
            for opp, val in sorted(original_vulnerabilities.items(), key=lambda kv: kv[1])
        ]
        remaining_list = [
            {"opponent": opp, "value": val}
            for opp, val in sorted(winner["residual_vulnerabilities"].items(), key=lambda kv: kv[1])
        ]
        worst = (
            {"opponent": winner["worst_residual_opponent"], "value": winner["worst_residual_value"]}
            if winner["worst_residual_opponent"] is not None
            else None
        )
        unknown_caveats = [
            {"opponent": opp, "backups_missing_data": sorted(missing)}
            for opp, missing in sorted(winner["unknown_relevant_opponents"].items())
        ]

        response = {
            "mode": mode,
            "main": result["main"],
            "secondary": result["secondary"],
            "role": role,
            "context": {"rank": rank, "input": input_value, "region": region},
            "original_vulnerability_count": len(original_vulnerabilities),
            "original_vulnerabilities": original_list,
            "final_residual_vulnerability_count": winner["residual_count"],
            "remaining_vulnerabilities": remaining_list,
            "vulnerabilities_resolved_count": len(winner["resolved_opponents"]),
            "vulnerabilities_resolved_opponents": winner["resolved_opponents"],
            "worst_remaining_matchup": worst,
            "no_known_unfavorable_matchups_remain": winner["residual_count"] == 0,
            "total_residual_severity": winner["total_residual_severity"],
            "total_matchup_improvement": winner["total_matchup_improvement"],
            "min_backup_wr": winner["min_backup_wr"],
            "mean_backup_wr": winner["mean_backup_wr"],
            "unknown_data_caveats": unknown_caveats,
            "search": {
                "candidate_pool_size": result["candidate_pool_size"],
                "pareto_frontier_size": result["pareto_frontier_size"],
            },
            "selection": {
                "decisive_tier": result["decisive_tier"],
                "worst_residual_value": winner["worst_residual_value"],
                "total_residual_severity": winner["total_residual_severity"],
                "min_backup_wr": winner["min_backup_wr"],
                "mean_backup_wr": winner["mean_backup_wr"],
            },
        }

        if mode == "builder":
            a, b = winner["backup_a"], winner["backup_b"]
            response["backup_a"] = a
            response["backup_b"] = b
            response["recommended_tertiary"] = None
            response["complete_pool"] = sorted([result["main"], a, b])
            response["contextual_wr"] = {"backup_a": winner["wr_a"], "backup_b": winner["wr_b"]}
        else:
            t = winner["tertiary"]
            response["backup_a"] = None
            response["backup_b"] = None
            response["recommended_tertiary"] = t
            response["complete_pool"] = sorted([result["main"], result["secondary"], t])
            response["contextual_wr"] = {"secondary": winner["wr_secondary"], "tertiary": winner["wr_tertiary"]}

        return response
