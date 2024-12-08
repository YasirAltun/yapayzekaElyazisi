from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Global değişken olarak modeli yükle
model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_image(image_data):
    """Gelen görüntüyü model için uygun formata dönüştürür"""
    # Base64 string'i görüntüye çevir
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
     
    # Görüntüyü 28x28 boyutuna getir
    image = image.resize((28, 28))
    
    # Numpy dizisine çevir ve normalize et
    image_array = np.array(image)
    image_array = image_array / 255.0
    
    # Model için uygun shape'e getir
    image_array = image_array.reshape(1, 28, 28)
    
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON verisini al
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Görüntüyü ön işle
        image_array = preprocess_image(data['image'])
        
        # Tahmin yap
        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        # Sonucu döndür
        return jsonify({
            'predicted_digit': predicted_class,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)