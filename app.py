from os import sep
from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = 'model_final.h5'
model = load_model(MODEL_PATH)
print("+"*50, "Model is loaded")

labels = pd.read_csv("labels.txt", sep="\n").values

@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (150,150))

	image = np.reshape(image, (1,150,150,3))

	pred = model.predict(image)

	pred = np.argmax(pred)

	labels = ["cataract", "normal"]

	pred = labels[pred]

	return render_template("prediction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)
