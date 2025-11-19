import io
import json
import numpy as np
from PIL import Image
import requests
from numpy import asarray
import os
import tensorflow as tf
import pathlib

SERVER_URL = 'http://localhost:8501/v1/models/flowers-model:predict'

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

if 'flower_photos' in os.listdir(data_dir):
    data_dir = os.path.join(data_dir, 'flower_photos')

folder_to_test = os.path.join(str(data_dir), 'tulips')
image_name = [f for f in os.listdir(folder_to_test) if f.endswith('.jpg')][0]
image_path = os.path.join(folder_to_test, image_name)

print(f"Probando con la imagen: {image_path}")

def main():
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print("Error: No se encontró la imagen.")
        return
    img = img.resize((64, 64))
    img_array = asarray(img)

    img_array = img_array.astype('float32') / 255.0

    img_list = np.expand_dims(img_array, 0).tolist()
    
    predict_request = json.dumps({'instances': img_list})

    print("\nEnviando petición al servidor...")
    try:
        response = requests.post(SERVER_URL, data=predict_request)
        
        if response.status_code != 200:
            print(f"\nERROR DEL SERVIDOR (Código {response.status_code}):")
            print(response.text)
            return

        response.raise_for_status()
        prediction = response.json()['predictions']
        
    except requests.exceptions.ConnectionError:
        print("Error: No se pudo conectar a Docker. Asegúrate de que el contenedor esté corriendo.")
        return

    print("\n--- Resultados Exitosos ---")
    classes_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    
    probs = prediction[0]
    idx_max = np.argmax(probs)
    image_output_class = classes_labels[idx_max]
    confidence = probs[idx_max] * 100

    print(f"Predicción: {image_output_class.upper()} con {confidence:.2f}% de confianza")

if __name__ == '__main__':
    main()