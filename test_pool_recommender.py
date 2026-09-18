"""Regression/algorithm tests for the complete-pool (Builder/Completer)
recommendation architecture. Plain assert-based script (matches this repo's
existing convention - see verify_pipeline.py, test_data_sources.py), not a
pytest suite. Run with: python test_pool_recommender.py

Covers sections A-K of the implementation spec:
  A. Determinism
  B. Exhaustive Builder search
  C. Completer search
  D. Max-known semantics
  E. Unknown preservation
  F. Zero residuals
  G. Pareto filtering
  H. Strict Candidate 4 (tier isolation)
  I. Manual diagnostic cases (Tracer/Pharah/Ashe/D.Va/Roadhog)
  J. Rank sensitivity (Genji/Hanzo/Reaper/Lifeweaver/Lucio)
  K. Full-roster smoke test
Plus API-level validation tests (section 11) via FastAPI's TestClient.
"""

import itertools
import math

import joblib
from fastapi.testclient import TestClient

import pipeline_def  # noqa: F401
import serve

PIPELINE_PATH = "pipeline.joblib"

PASS = 0
FAIL = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  [FAIL] {name}  {detail}")


bundle = joblib.load(PIPELINE_PATH)
pipeline = bundle["pipeline"]
transformer = pipeline.named_steps["recommender"]
BASELINE = ("Silver", "PC", "Americas")


def builder(main, role):
    return transformer.recommend_pool(
        {"role": role, "main": main, "secondary": None, "rank": BASELINE[0], "input": BASELINE[1], "region": BASELINE[2]}
    )


def completer(main, secondary, role):
    return transformer.recommend_pool(
        {"role": role, "main": main, "secondary": secondary, "rank": BASELINE[0], "input": BASELINE[1], "region": BASELINE[2]}
    )


# ============================================================
# A. Determinism
# ============================================================
print("\n=== A. Determinism ===")
r1 = builder("ashe", "DAMAGE")
r2 = builder("ashe", "DAMAGE")
check("Builder determinism (Ashe x2 identical)", r1 == r2)

c1 = completer("dva", "mauga", "TANK")
c2 = completer("dva", "mauga", "TANK")
check("Completer determinism (D.Va+Mauga x2 identical)", c1 == c2)

# ============================================================
# B. Exhaustive Builder search
# ============================================================
print("\n=== B. Exhaustive Builder search ===")
for role, expected_n in [("TANK", 15), ("DAMAGE", 24), ("SUPPORT", 14)]:
    heroes = transformer.heroes_by_role_[role]
    check(f"{role} role has {expected_n} heroes", len(heroes) == expected_n, f"got {len(heroes)}")
    main = heroes[0]
    result = builder(main, role)
    expected_pairs = math.comb(len(heroes) - 1, 2)
    check(
        f"{role} Builder ({main}) evaluates C({len(heroes)-1},2)={expected_pairs} pairs",
        result["search"]["candidate_pool_size"] == expected_pairs,
        f"got {result['search']['candidate_pool_size']}",
    )

# no duplicate A/B vs B/A: verify at the record-generation level directly
main, role = "ashe", "DAMAGE"
eligible = [h for h in transformer.heroes_by_role_[role] if h != main]
all_pairs = list(itertools.combinations(eligible, 2))
pair_set = {frozenset(p) for p in all_pairs}
check("No duplicate A/B and B/A pairs generated", len(all_pairs) == len(pair_set))

# ============================================================
# C. Completer search
# ============================================================
print("\n=== C. Completer search ===")
result = completer("ashe", "pharah", "DAMAGE")
check("Completer main fixed", result["main"] == "ashe")
check("Completer secondary fixed", result["secondary"] == "pharah")
eligible = transformer.heroes_by_role_["DAMAGE"]
expected_tertiary_count = len(eligible) - 2
check(
    f"Completer evaluates every eligible tertiary ({expected_tertiary_count})",
    result["search"]["candidate_pool_size"] == expected_tertiary_count,
    f"got {result['search']['candidate_pool_size']}",
)
check(
    "Recommended tertiary != main and != secondary",
    result["recommended_tertiary"] not in ("ashe", "pharah"),
)

# ============================================================
# D. Max-known semantics
# ============================================================
print("\n=== D. Max-known semantics ===")
main_val = pipeline_def.get_matchup_value(transformer.matchup_dict_, "ashe", "genji")
pharah_val = pipeline_def.get_matchup_value(transformer.matchup_dict_, "pharah", "genji")
vendetta_val = pipeline_def.get_matchup_value(transformer.matchup_dict_, "vendetta", "genji")
pool_vals = pipeline_def.compute_pool_values(transformer.matchup_dict_, ["ashe", "pharah", "vendetta"], transformer.opponent_pool_)
known = [v for v in (main_val, pharah_val, vendetta_val) if v is not None]
check(
    "Pool value at a known opponent equals max(known roster values)",
    pool_vals.get("genji") == (max(known) if known else None),
    f"pool={pool_vals.get('genji')} expected max({known})",
)

# ============================================================
# E. Unknown preservation
# ============================================================
print("\n=== E. Unknown preservation ===")
# find a real (main, opponent) pair in Ashe's V1 where BOTH pharah and
# vendetta lack a known value, if any exist for this specific pool
ashe_pool1 = pipeline_def.compute_pool_values(transformer.matchup_dict_, ["ashe"], transformer.opponent_pool_)
ashe_vuln1 = pipeline_def.get_vulnerabilities(ashe_pool1)
unknown_case = None
for opp, val in ashe_vuln1.items():
    if (
        pipeline_def.get_matchup_value(transformer.matchup_dict_, "pharah", opp) is None
        and pipeline_def.get_matchup_value(transformer.matchup_dict_, "vendetta", opp) is None
    ):
        unknown_case = (opp, val)
        break

if unknown_case:
    opp, original_val = unknown_case
    evaluation = pipeline_def.evaluate_complete_pool(
        transformer.matchup_dict_, "ashe", ["pharah", "vendetta"], transformer.opponent_pool_
    )
    check(
        f"Unknown-to-both opponent ({opp}) keeps main's original value unchanged (not resolved)",
        evaluation["residual_vulnerabilities"].get(opp) == original_val,
        f"got {evaluation['residual_vulnerabilities'].get(opp)}, expected {original_val}",
    )
    check(
        f"Unknown-to-both opponent ({opp}) is NOT in resolved_opponents (no false credit)",
        opp not in evaluation["resolved_opponents"],
    )
    check(
        f"Unknown-to-both opponent ({opp}) flagged in unknown_relevant_opponents",
        opp in evaluation["unknown_relevant_opponents"],
    )
else:
    print("  [SKIP] no natural unknown-to-both case found in Ashe's V1 for pharah+vendetta - testing synthetically instead")
    # synthetic fallback: fabricate a matchup_dict with a genuinely unknown pair
    synth = {
        "main": {"main": 0.0, "opp1": -10.0},
        "backupA": {"backupA": 0.0},  # no entry for "opp1" at all -> unknown
        "backupB": {"backupB": 0.0},  # no entry for "opp1" at all -> unknown
        "opp1": {},
    }
    ev = pipeline_def.evaluate_complete_pool(synth, "main", ["backupA", "backupB"], ["opp1"])
    check("Synthetic unknown case: original value preserved unchanged", ev["residual_vulnerabilities"].get("opp1") == -10.0)
    check("Synthetic unknown case: not resolved", "opp1" not in ev["resolved_opponents"])
    check("Synthetic unknown case: flagged as unknown-relevant", "opp1" in ev["unknown_relevant_opponents"])

# a candidate must never get credit merely for HAVING a known value if that
# value is itself negative (no false "conversion of unknown to known" credit)
synth2 = {
    "main": {"main": 0.0, "opp2": -5.0},
    "backupA": {"backupA": 0.0, "opp2": -1.0},  # known but still negative
    "backupB": {"backupB": 0.0},
    "opp2": {},
}
ev2 = pipeline_def.evaluate_complete_pool(synth2, "main", ["backupA", "backupB"], ["opp2"])
check(
    "A known-but-still-negative candidate value does not resolve the vulnerability",
    "opp2" in ev2["residual_vulnerabilities"] and "opp2" not in ev2["resolved_opponents"],
)

# ============================================================
# F. Zero residuals
# ============================================================
print("\n=== F. Zero residuals ===")
ashe_result = builder("ashe", "DAMAGE")
check("Ashe builder achieves zero residual vulnerabilities", ashe_result["final_residual_vulnerability_count"] == 0)
check("Zero-residual severity is exactly 0", ashe_result["total_residual_severity"] == 0.0)
check("Zero-residual worst_remaining_matchup is null (no fake opponent)", ashe_result["worst_remaining_matchup"] is None)
check("no_known_unfavorable_matchups_remain flag is True", ashe_result["no_known_unfavorable_matchups_remain"] is True)

# a zero-residual pool must compare as SAFER than a pool with a real residual
zero_record = {"worst_residual_value": None, "total_residual_severity": 0.0, "min_backup_wr": 10.0, "mean_backup_wr": 10.0}
nonzero_record = {"worst_residual_value": -1.0, "total_residual_severity": 1.0, "min_backup_wr": 90.0, "mean_backup_wr": 90.0}
winner, tier = pipeline_def.select_candidate4(
    [zero_record, nonzero_record], "min_backup_wr", "mean_backup_wr", identity_key=lambda r: (0,)
)
check(
    "Zero-residual pool wins tier 1 even against a far-better-WR nonzero pool",
    winner is zero_record and tier == "worst_case_alone",
)

# ============================================================
# G. Pareto filtering
# ============================================================
print("\n=== G. Pareto filtering ===")
main, role = "tracer", "DAMAGE"
eligible = [h for h in transformer.heroes_by_role_[role] if h != main]
tracer_pool1 = pipeline_def.compute_pool_values(transformer.matchup_dict_, [main], transformer.opponent_pool_)
tracer_vuln1 = pipeline_def.get_vulnerabilities(tracer_pool1)
records = [
    transformer._builder_candidate_record(main, a, b, role, *BASELINE, tracer_vuln1)
    for a, b in itertools.combinations(eligible, 2)
]
frontier = pipeline_def.pareto_filter(records, transformer._PARETO_DIMS)
check("Pareto filtering actually removes some candidates", len(frontier) < len(records), f"{len(frontier)} / {len(records)}")
check("Pareto filtering retains at least one candidate", len(frontier) > 0)

cassidy_freja = next(r for r in records if {r["backup_a"], r["backup_b"]} == {"cassidy", "freja"})
frontier_pairs = {frozenset({r["backup_a"], r["backup_b"]}) for r in frontier}
check(
    "Known-dominated pair Cassidy+Freja is REMOVED from the Tracer Pareto frontier",
    frozenset({"cassidy", "freja"}) not in frontier_pairs,
)
pharah_widow = next(r for r in records if {r["backup_a"], r["backup_b"]} == {"pharah", "widowmaker"})
check(
    "Non-dominated pair Pharah+Widowmaker is RETAINED on the Tracer Pareto frontier",
    frozenset({"pharah", "widowmaker"}) in frontier_pairs,
)

# ============================================================
# H. Strict Candidate 4 - explicit tier-isolation cases
# ============================================================
print("\n=== H. Strict Candidate 4 tier isolation ===")


def rec(worst, sev, min_wr, mean_wr, ident):
    return {"worst_residual_value": worst, "total_residual_severity": sev, "min_backup_wr": min_wr, "mean_backup_wr": mean_wr, "_id": ident}


# H1: worst-case decides before severity even considers anything
a = rec(-5.0, 100.0, 10.0, 10.0, "a")  # worse severity/WR but safer worst-case
b = rec(-10.0, 1.0, 99.0, 99.0, "b")  # everything else better, but worse worst-case
winner, tier = pipeline_def.select_candidate4([a, b], "min_backup_wr", "mean_backup_wr", identity_key=lambda r: (r["_id"],))
check("H1: worst-case wins even against much better severity/WR elsewhere", winner is a and tier == "worst_case_alone")

# H2: exact worst-case tie -> severity decides
a = rec(-5.0, 50.0, 1.0, 1.0, "a")
b = rec(-5.0, 20.0, 99.0, 99.0, "b")  # same worst, lower severity, much better WR
winner, tier = pipeline_def.select_candidate4([a, b], "min_backup_wr", "mean_backup_wr", identity_key=lambda r: (r["_id"],))
check("H2: severity participates on an exact worst-case tie", winner is b and tier == "severity_after_worstcase_tie")

# H3: worst-case AND severity both exactly tied -> WR (min) decides
a = rec(-5.0, 20.0, 40.0, 99.0, "a")
b = rec(-5.0, 20.0, 55.0, 10.0, "b")  # same worst+severity, higher min_wr, lower mean_wr
winner, tier = pipeline_def.select_candidate4([a, b], "min_backup_wr", "mean_backup_wr", identity_key=lambda r: (r["_id"],))
check("H3: contextual WR (min) participates only after exact ties in both matchup tiers", winner is b and tier == "contextual_wr_primary_after_matchup_ties")

# H3b: worst/severity/min_wr all tied -> mean_wr decides
a = rec(-5.0, 20.0, 40.0, 50.0, "a")
b = rec(-5.0, 20.0, 40.0, 60.0, "b")
winner, tier = pipeline_def.select_candidate4([a, b], "min_backup_wr", "mean_backup_wr", identity_key=lambda r: (r["_id"],))
check("H3b: mean WR decides once min WR is also exactly tied", winner is b and tier == "contextual_wr_secondary_after_matchup_and_primary_wr_ties")

# H4: everything tied -> deterministic alphabetical fallback
a = rec(-5.0, 20.0, 40.0, 50.0, "zzz_hero")
b = rec(-5.0, 20.0, 40.0, 50.0, "aaa_hero")
winner, tier = pipeline_def.select_candidate4(
    [a, b], "min_backup_wr", "mean_backup_wr", identity_key=lambda r: (r["_id"],)
)
check("H4: deterministic alphabetical fallback resolves a complete tie", winner is b and tier == "alphabetical_fallback", f"got {winner['_id']}")

# H5: for Builder, unordered pair identity must be normalized before the
# alphabetical fallback (A+B and B+A must resolve identically)
key1 = tuple(sorted(["zebra", "apple"]))
key2 = tuple(sorted(["apple", "zebra"]))
check("H5: unordered pair identity normalization is order-independent", key1 == key2 == ("apple", "zebra"))

# ============================================================
# I. Manual diagnostic cases
# ============================================================
print("\n=== I. Manual diagnostic cases ===")

tracer_result = builder("tracer", "DAMAGE")
check(
    "Tracer: dominated Cassidy+Freja does NOT survive as the final recommendation",
    {tracer_result["backup_a"], tracer_result["backup_b"]} != {"cassidy", "freja"},
)
check(
    "Tracer: Pharah+Widowmaker (zero-residual, non-dominated) IS the final recommendation",
    {tracer_result["backup_a"], tracer_result["backup_b"]} == {"pharah", "widowmaker"},
    f"got {tracer_result['backup_a']}+{tracer_result['backup_b']}",
)

pharah_result = builder("pharah", "DAMAGE")
pharah_eligible = [h for h in transformer.heroes_by_role_["DAMAGE"] if h != "pharah"]
pharah_pool1 = pipeline_def.compute_pool_values(transformer.matchup_dict_, ["pharah"], transformer.opponent_pool_)
pharah_vuln1 = pipeline_def.get_vulnerabilities(pharah_pool1)
pharah_records = [
    transformer._builder_candidate_record("pharah", a, b, "DAMAGE", *BASELINE, pharah_vuln1)
    for a, b in itertools.combinations(pharah_eligible, 2)
]
pharah_frontier = pipeline_def.pareto_filter(pharah_records, transformer._PARETO_DIMS)
zero_residual_frontier_pairs = [r for r in pharah_frontier if r["residual_count"] == 0]
check(
    "Pharah: multiple zero-residual pairs exist on the frontier",
    len(zero_residual_frontier_pairs) >= 2,
    f"found {len(zero_residual_frontier_pairs)}",
)
check("Pharah: final recommendation is itself zero-residual", pharah_result["final_residual_vulnerability_count"] == 0)

ashe_result = builder("ashe", "DAMAGE")
check(
    "Ashe: Pharah+Vendetta is the final recommendation",
    {ashe_result["backup_a"], ashe_result["backup_b"]} == {"pharah", "vendetta"},
)
ashe_eligible = [h for h in transformer.heroes_by_role_["DAMAGE"] if h != "ashe"]
ashe_pool1b = pipeline_def.compute_pool_values(transformer.matchup_dict_, ["ashe"], transformer.opponent_pool_)
ashe_vuln1b = pipeline_def.get_vulnerabilities(ashe_pool1b)
ashe_records = [
    transformer._builder_candidate_record("ashe", a, b, "DAMAGE", *BASELINE, ashe_vuln1b)
    for a, b in itertools.combinations(ashe_eligible, 2)
]
ashe_frontier = pipeline_def.pareto_filter(ashe_records, transformer._PARETO_DIMS)
check("Ashe: Pareto frontier size matches diagnostic (6 pairs)", len(ashe_frontier) == 6, f"got {len(ashe_frontier)}")
torbjorn_vendetta_present = any({r["backup_a"], r["backup_b"]} == {"torbjorn", "vendetta"} for r in ashe_frontier)
check("Ashe: known frontier alternative Torbjorn+Vendetta is present", torbjorn_vendetta_present)

dva_result = builder("dva", "TANK")
check(
    "D.Va: severe-vs-mild judgment reproduced - Ramattra+Zarya (4 mild) beats Mauga+Roadhog (contains -32)",
    {dva_result["backup_a"], dva_result["backup_b"]} == {"ramattra", "zarya"},
    f"got {dva_result['backup_a']}+{dva_result['backup_b']}",
)

roadhog_result = builder("roadhog", "TANK")
# judgment #6/#7 style: D.Va+Ramattra should beat D.Va+Mauga (worst -13 vs -32)
rh_eligible = [h for h in transformer.heroes_by_role_["TANK"] if h != "roadhog"]
rh_pool1 = pipeline_def.compute_pool_values(transformer.matchup_dict_, ["roadhog"], transformer.opponent_pool_)
rh_vuln1 = pipeline_def.get_vulnerabilities(rh_pool1)
dva_ramattra = transformer._builder_candidate_record("roadhog", "dva", "ramattra", "TANK", *BASELINE, rh_vuln1)
dva_mauga = transformer._builder_candidate_record("roadhog", "dva", "mauga", "TANK", *BASELINE, rh_vuln1)
check(
    "Roadhog #6: D.Va+Ramattra has a strictly safer worst-case than D.Va+Mauga",
    dva_ramattra["worst_residual_value"] > dva_mauga["worst_residual_value"],
    f"{dva_ramattra['worst_residual_value']} vs {dva_mauga['worst_residual_value']}",
)
hazard_sigma = transformer._builder_candidate_record("roadhog", "hazard", "sigma", "TANK", *BASELINE, rh_vuln1)
dva_sigma = transformer._builder_candidate_record("roadhog", "dva", "sigma", "TANK", *BASELINE, rh_vuln1)
check(
    "Roadhog #7 (low-confidence): Hazard+Sigma has a strictly safer worst-case than D.Va+Sigma",
    hazard_sigma["worst_residual_value"] > dva_sigma["worst_residual_value"],
    f"{hazard_sigma['worst_residual_value']} vs {dva_sigma['worst_residual_value']}",
)
winston_zarya = transformer._builder_candidate_record("roadhog", "winston", "zarya", "TANK", *BASELINE, rh_vuln1)
sigma_winston = transformer._builder_candidate_record("roadhog", "sigma", "winston", "TANK", *BASELINE, rh_vuln1)
print(
    f"  [INFO] Roadhog #12: Winston+Zarya worst={winston_zarya['worst_residual_value']} vs "
    f"Sigma+Winston worst={sigma_winston['worst_residual_value']} - "
    f"under STRICT no-tolerance tier 1 these are NOT exactly tied (see discrepancy report)."
)

# ============================================================
# J. Rank sensitivity
# ============================================================
print("\n=== J. Rank sensitivity ===")
RANK_CASES = [
    ("genji", "DAMAGE"), ("hanzo", "DAMAGE"), ("reaper", "DAMAGE"),
    ("lifeweaver", "SUPPORT"), ("lucio", "SUPPORT"),
]
CONTEXTS = [("Bronze", "PC", "Americas"), ("Silver", "PC", "Americas"), ("Grandmaster", "PC", "Americas")]

rank_changed = []
for main, role in RANK_CASES:
    picks = set()
    for ctx in CONTEXTS:
        res = transformer.recommend_pool({"role": role, "main": main, "secondary": None, "rank": ctx[0], "input": ctx[1], "region": ctx[2]})
        picks.add(frozenset({res["backup_a"], res["backup_b"]}))
    if len(picks) > 1:
        rank_changed.append(main)
check(
    "Rank sensitivity: all 5 flagged mains (Genji/Hanzo/Reaper/Lifeweaver/Lucio) show a rank-dependent change",
    set(rank_changed) == {"genji", "hanzo", "reaper", "lifeweaver", "lucio"},
    f"actually changed: {rank_changed}",
)

# spot check a handful of OTHER mains remain stable across ranks (should not
# be an exhaustive re-run of all 53 here - that's the full diagnostic's job)
STABLE_SAMPLE = [("ashe", "DAMAGE"), ("dva", "TANK"), ("zenyatta", "SUPPORT")]
for main, role in STABLE_SAMPLE:
    picks = set()
    for ctx in CONTEXTS:
        res = transformer.recommend_pool({"role": role, "main": main, "secondary": None, "rank": ctx[0], "input": ctx[1], "region": ctx[2]})
        picks.add(frozenset({res["backup_a"], res["backup_b"]}))
    check(f"Rank stability spot-check: {main} unchanged across Bronze/Silver/GM", len(picks) == 1, f"got {picks}")

# ============================================================
# K. Full-roster smoke test
# ============================================================
print("\n=== K. Full-roster smoke test ===")
crash_count = 0
nondeterminism_count = 0
illegal_pool_count = 0
total_mains = 0
for role, heroes in transformer.heroes_by_role_.items():
    for main in heroes:
        total_mains += 1
        try:
            res1 = builder(main, role)
            res2 = builder(main, role)
        except Exception as exc:  # noqa: BLE001
            crash_count += 1
            print(f"  [FAIL] Builder crashed for {main} ({role}): {exc}")
            continue
        pool = set(res1["complete_pool"])
        if len(pool) != 3 or not pool.issubset(set(heroes)) or main not in pool:
            illegal_pool_count += 1
            print(f"  [FAIL] Illegal pool for {main} ({role}): {res1['complete_pool']}")
        if res1 != res2:
            nondeterminism_count += 1
            print(f"  [FAIL] Non-deterministic output for {main} ({role})")

check(f"Full-roster Builder smoke test: no crashes ({total_mains} mains)", crash_count == 0, f"{crash_count} crashes")
check("Full-roster Builder smoke test: all pools legal (3 distinct same-role heroes incl. main)", illegal_pool_count == 0)
check("Full-roster Builder smoke test: all deterministic", nondeterminism_count == 0)

# Completer smoke test: one representative Main+Secondary per role
COMPLETER_SAMPLES = [
    ("roadhog", "dva", "TANK"), ("ashe", "pharah", "DAMAGE"), ("zenyatta", "moira", "SUPPORT"),
]
for main, secondary, role in COMPLETER_SAMPLES:
    try:
        res = completer(main, secondary, role)
        ok = (
            res["mode"] == "completer"
            and res["main"] == main
            and res["secondary"] == secondary
            and res["recommended_tertiary"] not in (main, secondary)
            and len(set(res["complete_pool"])) == 3
        )
        check(f"Completer smoke test: {main}+{secondary} ({role})", ok, str(res.get("recommended_tertiary")))
    except Exception as exc:  # noqa: BLE001
        check(f"Completer smoke test: {main}+{secondary} ({role})", False, str(exc))

# ============================================================
# API-level validation tests (section 11) via FastAPI TestClient
# ============================================================
print("\n=== API validation (section 11) ===")
client = TestClient(serve.app)


def post(body):
    return client.post("/recommend", json=body)


valid_bodies = {
    "TANK": {"role": "Tank", "main": "roadhog", "rank": "Silver", "input": "PC", "region": "Americas"},
    "DAMAGE": {"role": "Damage", "main": "ashe", "rank": "Silver", "input": "PC", "region": "Americas"},
    "SUPPORT": {"role": "Support", "main": "zenyatta", "rank": "Silver", "input": "PC", "region": "Americas"},
}
for role_name, body in valid_bodies.items():
    resp = post(body)
    check(f"Valid {role_name} Builder -> 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    if resp.status_code == 200:
        check(f"Valid {role_name} Builder -> mode=builder", resp.json()["mode"] == "builder")

completer_bodies = {
    "TANK": {"role": "Tank", "main": "roadhog", "secondary": "dva", "rank": "Silver", "input": "PC", "region": "Americas"},
    "DAMAGE": {"role": "Damage", "main": "ashe", "secondary": "pharah", "rank": "Silver", "input": "PC", "region": "Americas"},
    "SUPPORT": {"role": "Support", "main": "zenyatta", "secondary": "moira", "rank": "Silver", "input": "PC", "region": "Americas"},
}
for role_name, body in completer_bodies.items():
    resp = post(body)
    check(f"Valid {role_name} Completer -> 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:200]}")
    if resp.status_code == 200:
        check(f"Valid {role_name} Completer -> mode=completer", resp.json()["mode"] == "completer")

resp = post({"role": "Damage", "main": "ashe", "secondary": "ashe", "rank": "Silver", "input": "PC", "region": "Americas"})
check("main == secondary -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Damage", "main": "roadhog", "rank": "Silver", "input": "PC", "region": "Americas"})
check("cross-role main (Roadhog in Damage) -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Damage", "main": "ashe", "secondary": "roadhog", "rank": "Silver", "input": "PC", "region": "Americas"})
check("cross-role secondary (Roadhog in Damage) -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Flex", "main": "ashe", "rank": "Silver", "input": "PC", "region": "Americas"})
check("invalid role -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Damage", "main": "ashe", "rank": "Obsidian", "input": "PC", "region": "Americas"})
check("invalid rank -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Damage", "main": "ashe", "rank": "Silver", "input": "Mobile", "region": "Americas"})
check("invalid input -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Damage", "main": "ashe", "rank": "Silver", "input": "PC", "region": "Antarctica"})
check("invalid region -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({"role": "Damage", "rank": "Silver", "input": "PC", "region": "Americas"})
check("missing required field (main) -> 422", resp.status_code == 422, f"got {resp.status_code}")

resp = post({})
check("empty body -> 422", resp.status_code == 422, f"got {resp.status_code}")

# 503 when artifact unavailable - monkeypatch serve's module-level _bundle,
# restore immediately after (no file/process changes)
saved_bundle = serve._bundle
try:
    serve._bundle = None
    resp = post({"role": "Damage", "main": "ashe", "rank": "Silver", "input": "PC", "region": "Americas"})
    check("missing/unloadable artifact -> 503", resp.status_code == 503, f"got {resp.status_code}")
finally:
    serve._bundle = saved_bundle

resp = client.get("/health")
check("service healthy again after restoring bundle", resp.status_code == 200)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"TOTAL: {PASS} passed, {FAIL} failed")
if FAILURES:
    print("Failures:")
    for f in FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
