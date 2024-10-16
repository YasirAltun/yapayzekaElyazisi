import sys
import os
import openai
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageOps
import numpy as np
import base64

# OpenAI API anahtarınızı buraya ekleyin
openai.api_key = "openai api key"
# QT_PLUGIN_PATH ortam değişkenini manuel olarak ayarlayın
os.environ['QT_PLUGIN_PATH'] = '/Users/cybersurgeon/Desktop/yapayzekaElyazısı/.venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'


class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
    
    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()

        # 280x280 piksellik çizim alanı (10x scale for better drawing)
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(280, 280)
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Tahmin: ...')
        self.prediction.setFont(QtGui.QFont('Monospace', 20))

        self.button_clear = QtWidgets.QPushButton('Temizle')
        self.button_clear.clicked.connect(self.clear_canvas)

        self.button_predict = QtWidgets.QPushButton('Tahmin Et')
        self.button_predict.clicked.connect(self.predict)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_predict)

        self.setLayout(self.container)
    
    def clear_canvas(self):
        self.label.pixmap().fill(QtGui.QColor('black'))
        self.update()

    def predict(self):
        # Resmi kaydet
        self.label.pixmap().save('temp.png', 'PNG')
        
        # Resmi yükle
        img = Image.open('temp.png')
        
        # Resmi gri tonlamaya çevir ve yeniden boyutlandır
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Resmi numpy array'e çevir ve normalize et
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)
        
        # Resmi base64 formatına çevir
        img_bytes = img_array.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # OpenAI API ile tahmin et
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an digit recognition AI that identifies digits."},
                {"role": "user", "content": f"This is the base64-encoded image data: {img_b64}. What digit is in the image?"}
            ]
        )
        
        # Tahmini al ve göster
        predicted_class = response.choices[0].message['content'].strip()
        self.prediction.setText(f'Tahmin: {predicted_class}')
    
    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        painter = QtGui.QPainter(self.label.pixmap())
        p = painter.pen()
        p.setWidth(10)  # 10x scale for better drawing
        self.pen_color = QtGui.QColor('white')
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)
        self.setWindowTitle('RAKAM TAHMİNİ')
        self.setGeometry(100, 100, 300, 400)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    mainApp = MainWindow()
    mainApp.show()
    sys.exit(app.exec_())
