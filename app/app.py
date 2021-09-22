from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
# def hello_world():
#     model = load("../model/model.joblib")
#     test_input = np.array([[1], [2], [12]])
#     preds = model.predict(test_input)
#     return str(preds)
def index():
    if request.method == "GET":
        return render_template('index.html', href='static/base-pic.svg')
    else:
        text = request.form['ages']
        pic_path = 'static/predictions_pic.svg'
        model = load('model.joblib')
        new_input = floats_str_to_np_array(text)
        make_picture('AgesAndHeights.pkl', model, new_input, pic_path)

        return render_template('index.html', href=pic_path)


def make_picture(training_data_filename, model, new_input_np, output_file):
    data = pd.read_pickle(training_data_filename)
    data["Height_cm"] = data["Height"] * 2.54

    data = data[data.Age > 0]

    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)

    fig = px.scatter(data_frame=data, x="Age", y="Height_cm", 
                    title="Height vs. Age", 
                    labels={'Age':'Ages (years)', 'Height_cm':'Height (cm)'})

    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

    # add marker for new input

    new_preds = model.predict(new_input_np)
    fig.add_trace(go.Scatter(x=new_input_np.reshape(len(new_input_np)), 
                             y=new_preds, name="New Output", 
                             mode='markers', marker=dict(color='purple', size=16)))

    fig.write_image(output_file, width=800, engine='kaleido')
    # fig.show()


def floats_str_to_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)












if __name__ == "__main__":
    app.run(debug=True)