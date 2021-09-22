from flask import Flask
import numpy as np
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
    model = load("../model/model.joblib")
    test_input = np.array([[1], [2], [12]])
    preds = model.predict(test_input)
    return str(preds)


@app.route("/blog")
def blog():
    return """<h1>Блог</h1>
    <p>Hello, World!</p>
    <p>Привет привет</p>
    """


if __name__ == "__main__":
    app.run(debug=True)