from unittest import result
import bentoml
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("credit_risk_model:latest")
# credit_risk_model:iusktlssmgtxfshc -- latest chooses teh most recent tag
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service(
    'credit_risk_classifier', runners=[model_runner]
)


@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)

    result = prediction[0]

    if result > 0.5:
        return {"status": "Declined"}
    elseif result > 0.25:
        return {"status": 'Approved'}
    return {"status": 'Approved'}
