import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle




app = Flask(__name__)
model = pickle.load(open('obesity_grad_model.bst', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    independent_set = [i for i in request.form.values()]
    independent_set = np.array(independent_set)
    independent_set = independent_set.reshape(1, -1)

    i = model.predict(independent_set)
    pred_dict = {'Insufficient weight': 0, 'Normal weight': 1, 'Overweight level I': 2, 'Overweight Level II': 3,
                 'Obesity_Type I': 4, 'Obesity Type II': 5, 'Obesity Type III': 6}
    pred_list = list(pred_dict)
    prediction = pred_list[int(i)]
    return render_template("index.html", prediction_text=f'You belong to the {prediction} category')


if __name__ == "__main__":
    app.run(debug=True)
