"""Load pipeline.joblib fresh from disk and verify the OLD sequential
Formula C path (HeroRecommenderTransformer._recommend_for_request) still
produces the already-validated results, unchanged.

Note: pipeline.transform() itself now dispatches to the NEW complete-pool
Builder/Completer architecture (see test_pool_recommender.py for its
regression suite) - this script deliberately calls the legacy sequential
method directly so the old behavior remains comparable against the new
architecture, per the post-submission-development preservation requirement."""

import joblib

import pipeline_def  # noqa: F401  (must be importable before unpickling the bundle)

PIPELINE_PATH = "pipeline.joblib"


def summarize(result):
    picks = [r["recommended_hero"] for r in result["recommendations"]]
    return f"input_heroes={result['input_heroes']} -> recommendations={picks} final_roster={result['final_roster']}"


def main():
    bundle = joblib.load(PIPELINE_PATH)
    pipeline = bundle["pipeline"]
    transformer = pipeline.named_steps["recommender"]

    print(f"Loaded bundle. Metadata: {bundle['metadata']}")

    requests = [
        {"heroes": ["soldier-76"], "role": "DAMAGE", "rank": "Silver", "input": "PC", "region": "Americas"},
        {"heroes": ["tracer"], "role": "DAMAGE", "rank": "Silver", "input": "PC", "region": "Americas"},
        {"heroes": ["soldier-76", "symmetra"], "role": "DAMAGE", "rank": "Silver", "input": "PC", "region": "Americas"},
    ]

    output = [transformer._recommend_for_request(r) for r in requests]

    print(f"\ntransform() return type: {type(output)}")
    print(f"transform() return length: {len(output)}")
    print(f"Element type: {type(output[0])}")
    print(f"Element keys: {list(output[0].keys())}")
    print(f"recommendations[0] keys: {list(output[0]['recommendations'][0].keys())}")

    print("\n--- Results ---")
    for result in output:
        print(summarize(result))

    # --- Assertions against already-verified sequential results ---
    soldier_result = output[0]
    tracer_result = output[1]
    two_hero_result = output[2]

    soldier_picks = [r["recommended_hero"] for r in soldier_result["recommendations"]]
    assert soldier_picks == ["symmetra", "widowmaker"], f"soldier-76 mismatch: {soldier_picks}"

    tracer_picks = [r["recommended_hero"] for r in tracer_result["recommendations"]]
    assert tracer_picks == ["cassidy", "freja"], f"tracer mismatch: {tracer_picks}"

    assert len(two_hero_result["recommendations"]) == 1, (
        f"two-hero input should yield exactly 1 recommendation, got "
        f"{len(two_hero_result['recommendations'])}"
    )
    two_hero_pick = two_hero_result["recommendations"][0]["recommended_hero"]
    assert two_hero_pick == "widowmaker", f"two-hero slot #3 mismatch: {two_hero_pick}"

    print("\nAll assertions passed:")
    print(f"  soldier-76 sequential picks: {soldier_picks} (expected [symmetra, widowmaker])")
    print(f"  tracer sequential picks:     {tracer_picks} (expected [cassidy, freja])")
    print(f"  [soldier-76, symmetra] slot #3 pick: {two_hero_pick} (expected widowmaker, count=1)")


if __name__ == "__main__":
    main()
