import os
from flask import Flask, request, jsonify
import pickle
import numpy as np
import warnings
warnings.simplefilter('ignore')

# загружаем модель из файла
with open(os.path.join('models', 'random_forest.pkl'), 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Тестовое сообщение. Сервер запущен!"
    return msg

# создаем ендпоинт с помощью которого будем получать предикт от модели
@app.route('/predict', methods=['POST'])
def predict():
    features = np.array(request.json)
    print(features.shape)
    prediction = np.round(np.exp(model.predict(features))).tolist()
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run('localhost', 5000)