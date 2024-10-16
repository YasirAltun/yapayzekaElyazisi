import os
import sys
import json
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# QT_PLUGIN_PATH ortam değişkenini manuel olarak ayarlayın
os.environ['QT_PLUGIN_PATH'] = '/Users/cybersurgeon/Desktop/yapayzekaElyazısı/.venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'

class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        # Eğitilmiş modeli yükle
        self.loaded_model = tf.keras.models.load_model('mnist_model.h5')

        # Doğruluk oranını yükle
        with open('accuracy.json', 'r') as f:
            self.accuracy = json.load(f)['accuracy']

        self.initUI()
    
    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()

        # 28x28 piksellik çizim alanı
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(280, 280)  # 10x scale for better drawing
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Tahmin: ...')
        self.prediction.setFont(QtGui.QFont('Monospace', 20))

        self.button_clear = QtWidgets.QPushButton('Temizle')
        self.button_clear.clicked.connect(self.clear_canvas)

        self.button_predict = QtWidgets.QPushButton('Tahmin Et')
        self.button_predict.clicked.connect(self.predict)

        self.accuracy_label = QtWidgets.QLabel(f'Doğruluk Oranı: {self.accuracy * 100:.2f}%')
        self.accuracy_label.setFont(QtGui.QFont('Monospace', 15))

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_predict)
        self.container.addWidget(self.accuracy_label, alignment=QtCore.Qt.AlignHCenter)

        self.setLayout(self.container)
    
    def clear_canvas(self):
        self.label.pixmap().fill(QtGui.QColor('black'))
        self.update()

    def predict(self):
        s = self.label.pixmap().toImage().bits().asstring(280 * 280 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((280, 280, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28, 28), Image.Resampling.LANCZOS)))
        arr = (arr / 255.0).reshape(1, 28, 28)

        prediction = self.loaded_model.predict(arr)
        predicted_class = np.argmax(prediction, axis=1)[0]

        self.prediction.setText('Tahmin: ' + str(predicted_class))
    
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
