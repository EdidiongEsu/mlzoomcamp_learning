import numpy as np

import bentoml
from bentoml.io import JSON

# how the teag was created in in train notebook in the bentoml section
model_ref = bentoml.sklearn.load_model("decision_tree:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = bentoml.sklearn.get("decision_tree:latest").to_runner()

svc = bentoml.Service("decision_tree", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    result = prediction[0]

    if result > 0.5:
        return {
            "status": "DECLINED"
        }
    elif result > 0.25:
        return {
            "status": "MAYBE"
        }
    else:
        return {
            "status": "APPROVED"
        }
