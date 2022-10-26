import bentoml
from bentoml.io import JSON, NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:latest")

model_runner = model_ref.to_runner()

svc = bentoml.Service('q5_service', runners = [model_runner])

@svc.api(input = NumpyNdarray(), output = JSON())
async def classify(input_data):
    prediction = await model_runner.predict.async_run(input_data)
    print(prediction)

    result = prediction[0]
    
    return {"result":result}