from keras_image_helper import create_preprocessor
import tflite_runtime.interpreter as tflite

path = 'Dataset/test/Hyundai Creta/88.jpg'


def predict(path):
    classifier_model = "car-model.tflite"
    target_size = (299, 299)

    interpreter = tflite.Interpreter(model_path=classifier_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    preprocessor = create_preprocessor('xception', target_size=target_size)

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

    results = dict(zip(classes, preds[0]))

    print(results)
    return results


predict(path=path)
