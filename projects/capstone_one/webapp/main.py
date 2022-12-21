import streamlit as st
import pandas as pd
from keras_image_helper import create_preprocessor
import tensorflow.lite as tflite


import tflite_runtime.interpreter as tflite

# st.set_page_config(layout="wide")

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Car Image Prediction service')
st.header(
    "Welcome to this web application that classifies Cars. There are six classes of Cars- [Audi',Hyundai Creta',Mahindra Scorpio',Rolls Royce',Swift',Tata Safari, Toyota Innova]")


df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

df

def predict(image):
    classifier_model = "car-model.tflite"
    target_size = (299, 299)
    
    interpreter = tflite.Interpreter(model_path= classifier_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    preprocessor = create_preprocessor('xception', target_size=target_size)

    path = 'Dataset/test/Hyundai Creta/88.jpg'
    X = preprocessor.from_path(path)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    classes = [
        'Audi',
        'Hyundai Creta',
        'Mahindra Scorpio',
        'Rolls Royce',
        'Swift',
        'Tata Safari',
        'Toyota Innova'
    ]

    dict(zip(classes, preds[0]))