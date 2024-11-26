from flask import Flask, request, jsonify

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import *
import base64
from PIL import Image
from io import BytesIO

# from sklearn.ensemble import RandomForestClassifier
# import joblib

app = Flask(__name__)

# carrega o modelo
# aqui vocês precisam criar a estrutura de modelo de vocês
# ex.

# Função personalizada de pré-processamento
def preprocess_image(image):
    """
    Função para pré-processar a imagem recebida.
    - Recebe a imagem no formato PIL.
    - Redimensiona, converte para array e aplica o pré-processamento necessário.
    """
    try:
        # Redimensiona e converte para array
        image = image.resize((img_height,img_width))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # Normaliza os valores de entrada para o intervalo adequado
        image = preprocess_input(image)
    except Exception as e:
        raise ValueError(f"Erro ao pré-processar a imagem: {e}")
    return image


def format_response(predictions):
    """
    Realiza a predição e formata a resposta.
    - Recebe a imagem já pré-processada como input.
    - Retorna a classe e a probabilidade associada.
    """
    try:
        # Formata a resposta (substitua as labels conforme as classes do seu dataset)
        class_labels = ["cardboard","glass","metal","paper","plastic","trash"]
        response = [
            {'classe': class_labels[i], 'probabilidade': f"{prob * 100:.2f} %"} 
            for i, prob in enumerate(predictions[0])
        ]
        # Ordena pela probabilidade em ordem decrescente
        response = sorted(response, key=lambda x: float(x['probabilidade'].split()[0]), reverse=True)
    except Exception as e:
        raise ValueError(f"Erro ao realizar a inferência: {e}")
    return response

img_height, img_width = 224, 224
num_classes = 6

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# congela as camadas convolucionais para preservar os pesos pré-treinados
for layer in vgg_base.layers:
    layer.trainable = False

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# carrega os pesos para o modelo
model.load_weights('artefatos/modeloIALixo.weights.h5')

@app.route("/predict", methods=['POST'])
def predict():

    dados = request.json

    if dados ==  None:
        return jsonify("Nenhuma imagem foi enviada")
    
    try:
        img_data = base64.b64decode(dados['imagem'])
        img = Image.open(BytesIO(img_data)).convert('RGB')
    except Exception as e:
        return jsonify("Erro ao processar a imagem. Erro -> " + str(e))
    
    try:
        preprocessed_image = preprocess_image(img)

        prediction = model.predict(preprocessed_image)

        response = format_response(prediction)
    
    except Exception as e:
        return jsonify("Erro ao inferir o resultado. Erro -> " + str(e))

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)