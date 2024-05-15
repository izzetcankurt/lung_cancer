from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
import sys
import predict

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        
        # Load the UI file
        uic.loadUi("predesign.ui", self)
        
        # Define our widgets
        # (Example: self.my_button = self.findChild(QPushButton, "myButton"))
        self.button = self.findChild(QPushButton, "pushButton")
        self.label = self.findChild(QLabel, "label")
        self.label2 = self.findChild(QLabel, "label_2")
        
        self.button.clicked.connect(self.clicker)
        
        # Show the App
        self.show()
    
    def clicker(self):
        # self.label.setText("Clicked")
        fname, _ = QFileDialog.getOpenFileName(self, "Open File")
        if fname:
            self.label.setText(str(fname))
            pred = predict.prediction(str(fname))
            self.label2.setText(pred)
    
# Initialize the App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
