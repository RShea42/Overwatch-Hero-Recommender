"""Build and serialize the sklearn pipeline bundle for the hero recommender."""

import platform
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from pipeline_def import HeroRecommenderTransformer

MATCHUPS_PATH = "data/matchups.csv"
RANK_STATS_PATH = "data/rank_stats.csv"
OUTPUT_PATH = "pipeline.joblib"


def build_bundle():
    matchups_df = pd.read_csv(MATCHUPS_PATH)
    rank_df = pd.read_csv(RANK_STATS_PATH)

    pipeline = Pipeline(
        steps=[
            ("recommender", HeroRecommenderTransformer(matchups_df=matchups_df, rank_df=rank_df)),
        ]
    )

    # fit() derives and stores the symmetric matchup matrix and hero pools as
    # genuine fitted (trailing-underscore) attributes on the transformer -
    # this is what gets serialized below, not recomputed at request time.
    pipeline.fit(X=[])

    heroes_by_role = {
        role: sorted(group["hero_id"].unique()) for role, group in rank_df.groupby("role")
    }
    all_heroes = sorted(set(matchups_df["hero_id"]) | set(matchups_df["opponent_id"]))

    bundle = {
        "pipeline": pipeline,
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
        "heroes_by_role": heroes_by_role,
        "all_heroes": all_heroes,
    }

    return bundle


def main():
    bundle = build_bundle()
    joblib.dump(bundle, OUTPUT_PATH)

    print(f"Saved bundle to {OUTPUT_PATH}")
    print(f"sklearn version: {sklearn.__version__}")
    print(f"Pipeline steps: {bundle['metadata']['steps']}")
    print(f"Bundle keys: {list(bundle.keys())}")
    print(f"Metadata: {bundle['metadata']}")
    for role, heroes in bundle["heroes_by_role"].items():
        print(f"{role} heroes tracked: {len(heroes)}")
    print(f"All heroes tracked: {len(bundle['all_heroes'])}")


if __name__ == "__main__":
    main()
