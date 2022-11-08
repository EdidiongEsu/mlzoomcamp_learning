import numpy as np

import bentoml
from bentoml.io import JSON

# how the teag was created in in train notebook in the bentoml section

model_ref = bentoml.sklearn.get("decision_tree:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = bentoml.sklearn.get("decision_tree:latest").to_runner()

svc = bentoml.Service("decision_tree", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    vector = dv.transform(application_data)
    prediction = await model_runner.predict_proba.async_run(vector)
    print(prediction)
    result = prediction[0][1]
    rounded_result = f"{result:.2%}"
    print(result)

    if result >= 0.5:
        return {
            "Probability of churning": rounded_result,
            "status": "Customer will churn"
        }
    else:
        return {
            "Probability of churning": rounded_result,
            "status": "Customer will not churn"
        }
