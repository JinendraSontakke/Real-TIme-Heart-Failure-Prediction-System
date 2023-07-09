from flask import Flask, render_template, request
from model import Model
import pandas as pd
import time
import pickle

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            file.save('prediction.csv')
            data = pd.read_csv('prediction.csv')
            age = data['Age']
            heart_rate = data['heart Rate (bpm)']
            
            with open('model1.pkl', 'rb') as model_file:
                model = pickle.load(model_file)

            
            #predictions = model.predict([[age, heart_rate]])
            predictions = model.predict(age, heart_rate)
            if predictions == 0:
                condition = 'Good'
            elif predictions == 1:
                condition = 'Moderate'
            else:
                condition = 'Warning'

            time.sleep(5)

            return render_template('result.html', condition=condition)

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
