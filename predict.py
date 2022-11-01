import numpy as np
from flask import Flask, request, render_template, app
import pickle
app = Flask(__name__)

def ValuePredictor(to_predict_list):
    print("inside func")
    to_predict = np.array(to_predict_list).reshape(1, 17)
    loaded_model = pickle.load(open("RFR.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        print("inside predict")
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        return render_template("result.html", prediction=result)

