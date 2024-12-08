import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps

app = Flask(__name__)

# Model eğitimi ve kaydedilmesi
def train_and_save_model(model_path):
    from tensorflow.keras.datasets import mnist

    # MNIST veri kümesini yükle
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Veriyi normalize et
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Modeli oluştur
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Modeli derle
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Modeli eğit
    model.fit(x_train, y_train, epochs=10)

    # Test doğruluğunu kontrol et
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Doğruluğu: {test_acc}")

    # Modeli kaydet
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}")

# Modeli yükle
model_path = './mnist_model.h5'
if not os.path.exists(model_path):
    print("Model bulunamadı, eğitim süreci başlatılıyor...")
    train_and_save_model(model_path)

model = tf.keras.models.load_model(model_path)

# Görsel işleme fonksiyonu
def preprocess_image(image, target_size=(28, 28)):
    img = image.convert('L')  # Grayscale
    img = img.resize(target_size, Image.Resampling.LANCZOS)  # Yeniden boyutlandır
    img_array = np.array(img) / 255.0  # Normalize et
    input_array = img_array.reshape(1, target_size[0], target_size[1], 1)  # Model girişine uygun hale getir
    return input_array

# Flask rotaları
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400

    file = request.files['file']
    try:
        img = Image.open(file)
        input_array = preprocess_image(img)
        prediction = model.predict(input_array)
        digit = int(np.argmax(prediction, axis=1)[0])
        return jsonify({'prediction': digit})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HTML içeriği
os.makedirs('templates', exist_ok=True)
html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MNIST Digit Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        #imagePreview {
            max-width: 300px;
            max-height: 300px;
            margin: 20px auto;
            border: 2px dashed #ccc;
        }
        #predictionResult {
            font-size: 24px;
            margin: 20px 0;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>MNIST Digit Predictor</h1>
    <input type="file" id="imageUpload" accept="image/*">
    <img id="imagePreview" style="display:none;">
    <div id="predictionResult"></div>

    <script>
        const imageUpload = document.getElementById('imageUpload');
        const imagePreview = document.getElementById('imagePreview');
        const predictionResult = document.getElementById('predictionResult');

        imageUpload.addEventListener('change', function(event) {
            const file = event.target.files[0];
            const reader = new FileReader();

            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };

            reader.readAsDataURL(file);
        });

        imageUpload.addEventListener('change', function(event) {
            const file = event.target.files[0];
            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.prediction !== undefined) {
                    predictionResult.textContent = `Tahmin Edilen Rakam: ${data.prediction}`;
                } else if (data.error) {
                    predictionResult.textContent = `Hata: ${data.error}`;
                }
            })
            .catch(error => {
                predictionResult.textContent = 'Bir hata oluştu.';
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
'''
with open('templates/index.html', 'w') as f:
    f.write(html_content)

# Uygulamayı başlat
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
