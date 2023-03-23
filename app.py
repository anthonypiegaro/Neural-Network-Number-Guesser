from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
from keras.models import model_from_json

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model/model.h5")
print("Loaded model from disk")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_number():
    data = request.get_json()

    array = np.array(data["data"])

    digit = np.argmax(model.predict(array.reshape((1, 28, 28, 1)))[0], axis=-1)

    res = make_response(jsonify({"message": str(digit)}), 200)

    return res
