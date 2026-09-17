"""Modal deployment of the hero-recommendation FastAPI service.

This file only builds the container image and exposes the existing FastAPI
app (serve.py) as a Modal web function. It does not reimplement anything:
the recommendation algorithm, the pipeline artifact, and the API route
definitions are all unchanged and just shipped into the image as-is.

Local files shipped into the image - exactly the three application files
required (pipeline_def.py is self-contained: the matchup-matrix/pool/scoring
helpers live there directly, so there is no runtime import of
explore_recommendations.py):
  - pipeline.joblib   the fitted sklearn pipeline bundle
  - pipeline_def.py   defines HeroRecommenderTransformer and the pure
                      scoring helpers it uses (needed to unpickle and run it)
  - serve.py          the FastAPI app itself

data/matchups.csv and data/rank_stats.csv are deliberately NOT shipped:
pipeline.joblib already contains the fitted transformer's matchups_df and
rank_df as pickled attributes (they were passed into
HeroRecommenderTransformer.__init__ and stored on self at build time), so
the deployed runtime never touches the CSVs. This was confirmed locally by
temporarily moving data/ aside and re-running verify_pipeline.py against
pipeline.joblib alone - it passed unchanged.
"""

import sys

import modal

app = modal.App("overwatch-hero-recommender")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # Pinned to exactly match this project's local venv, so the
        # deployed container is running byte-identical dependency versions
        # to the ones pipeline.joblib was built and verified against.
        "scikit-learn==1.9.1",
        "pandas==3.0.5",
        "numpy==2.5.3",
        "joblib==1.6.0",
        "fastapi==0.141.1",
        "pydantic==2.13.5",
        # uvicorn is not imported by serve.py or pipeline_def.py - Modal
        # serves the ASGI app itself via @modal.asgi_app(), so uvicorn is not
        # actually required at runtime here. Pinned anyway per explicit
        # instruction; harmless to include.
        "uvicorn==0.53.0",
    )
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib", copy=True)
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py", copy=True)
    .add_local_file("serve.py", "/root/serve.py", copy=True)
)


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    # Imported here, inside the remote function body, so this import runs
    # inside the Modal container (where /root/pipeline.joblib is the
    # deployed artifact) rather than on the local host machine when this
    # module is loaded by the `modal deploy` CLI - which would otherwise
    # import serve.py locally and load the LOCAL pipeline.joblib instead.
    sys.path.insert(0, "/root")
    from serve import app as web_app

    return web_app
