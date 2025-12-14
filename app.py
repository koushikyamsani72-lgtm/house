
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    prediction = model.predict([values])[0]
    return render_template("result.html", prediction=round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True)
