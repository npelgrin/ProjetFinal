# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Analyse.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import Prediction

class Ui_AnalyseUI(object):
    def setupUi(self, AnalyseUI):
        AnalyseUI.setObjectName("AnalyseUI")
        AnalyseUI.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(AnalyseUI)
        self.centralwidget.setObjectName("centralwidget")
        self.ButtonSelect = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonSelect.setGeometry(QtCore.QRect(80, 140, 75, 23))
        self.ButtonSelect.setObjectName("ButtonSelect")
        self.ButtonAnalyse = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonAnalyse.setGeometry(QtCore.QRect(80, 430, 75, 23))
        self.ButtonAnalyse.setObjectName("ButtonAnalyse")
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setGeometry(QtCore.QRect(350, 20, 331, 281))
        self.Image.setFrameShape(QtWidgets.QFrame.Box)
        self.Image.setText("")
        self.Image.setObjectName("Image")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(350, 340, 331, 181))
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setText("")
        self.label.setObjectName("label")
        AnalyseUI.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AnalyseUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.menubar.setFont(font)
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        AnalyseUI.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AnalyseUI)
        self.statusbar.setObjectName("statusbar")
        AnalyseUI.setStatusBar(self.statusbar)

        self.retranslateUi(AnalyseUI)
        QtCore.QMetaObject.connectSlotsByName(AnalyseUI)
        
        self.ButtonSelect.clicked.connect(self.setImage)
        
        self.ButtonAnalyse.clicked.connect(self.printResultats)


    def retranslateUi(self, AnalyseUI):
        _translate = QtCore.QCoreApplication.translate
        AnalyseUI.setWindowTitle(_translate("AnalyseUI", "MainWindow"))
        self.ButtonSelect.setText(_translate("AnalyseUI", "Select image"))
        self.ButtonAnalyse.setText(_translate("AnalyseUI", "Analyser"))
        
    def setImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
        if fileName: # If the user gives a file
            path = open('chemin.txt','w')
            path.write(fileName)
            path.close()
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.Image.width(), self.Image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.Image.setPixmap(pixmap) # Set the pixmap onto the label
            self.Image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
            
    def printResultats(self):
        ipath = open('chemin.txt','r')
        adresse = ipath.read()
        ipath.close()
        Prediction.main(adresse)
        Fichier = open('predictions.txt', 'r')
        chaine = Fichier.read()
        self.label.setText('Contenu du fichier :\n' + chaine)
        Fichier.close()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AnalyseUI = QtWidgets.QMainWindow()
    ui = Ui_AnalyseUI()
    ui.setupUi(AnalyseUI)
    AnalyseUI.show()
    sys.exit(app.exec_())

