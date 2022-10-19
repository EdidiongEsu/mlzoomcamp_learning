# Load the model
import pickle

from flask import Flask
from flask import request
from flask import jsonify


def loadfile(filename):
    with open(filename, "rb") as f:
        output = pickle.load(f)
    return output


model = loadfile("model1.bin")
dv = loadfile("dv.bin")

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
