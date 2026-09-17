"""Post-submission DEVELOPMENT Modal deployment - separately named from the
submitted app so it never overwrites the graded deployment.

Identical to modal_serve.py except for the App name below. The submitted
app ("overwatch-hero-recommender", deployed via modal_serve.py) must never
be redeployed during post-submission development; this file is the only
thing `modal deploy` should be pointed at while iterating.

Local files shipped into the image - exactly the three application files
(pipeline_def.py is self-contained, no runtime import of
explore_recommendations.py):
  - pipeline.joblib   the fitted sklearn pipeline bundle
  - pipeline_def.py   defines HeroRecommenderTransformer and the pure
                      scoring helpers it uses (needed to unpickle and run it)
  - serve.py          the FastAPI app itself
"""

import sys

import modal

app = modal.App("overwatch-hero-recommender-dev")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "scikit-learn==1.9.1",
        "pandas==3.0.5",
        "numpy==2.5.3",
        "joblib==1.6.0",
        "fastapi==0.141.1",
        "pydantic==2.13.5",
        "uvicorn==0.53.0",
    )
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib", copy=True)
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py", copy=True)
    .add_local_file("serve.py", "/root/serve.py", copy=True)
)


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, "/root")
    from serve import app as web_app

    return web_app
