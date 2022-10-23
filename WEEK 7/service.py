from email.mime import application
from unittest import result

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

import pydantic
from pydantic import BaseModel

import numpy as np


class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital:  str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int


model_ref = bentoml.xgboost.get("credit_risk_model:latest")
# credit_risk_model:iusktlssmgtxfshc -- latest chooses teh most recent tag
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service(
    'credit_risk_classifier', runners=[model_runner]
)


@svc.api(input= NumpyNdarray(shape=(-1,29), dtype=np.float32,enforce_dtype=True,enforce_shape = True), output=JSON())
def classify(vector):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)

    result = prediction[0]

    if result > 0.5:
        return {"status": "Declined"}
    elif result > 0.2:
        return {"status": 'Maybe'}
    else:
        return {"status": 'Approved'}
