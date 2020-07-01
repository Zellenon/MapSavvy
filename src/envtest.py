import sys
from PySide2.QtCore import QThread, QThreadPool, Signal
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout
import numpy as np
import numpy.random as nprand
from timeit import default_timer as time

class MyThread(QThread):

	updateProgress = Signal(float)

	def __init__(self):
		QThread.__init__(self)

	def run(self):
		n = 2000
		a = nprand.randint(-10,10,n)
		b = np.zeros((n,n))
		for i in range(0,len(a)):
			for j in range(0,len(a)):
				b[i][j] = a[i] * a[j]
			if (i % (n//100) == 0):
				self.updateProgress.emit(i / (n//100))
		self.updateProgress.emit(100)

def timedFunction():
	a = nprand.randint(-10,10,10000)
	b = np.zeros((10000,10000))
	for i in range(0,len(a)):
		for j in range(0,len(a)):
			b[i][j] = a[i] * a[j]
		if (i % 100 == 0):
			print (i)
	print(len(a))

def buttonPressed():
	s = time()
	timedFunction()
	e = time()
	print('Time: ' + str(e-s))

# Qt Application
app = QApplication(sys.argv)
# QWidget
widget = QWidget()
mainLayout = QVBoxLayout()
startButton = QPushButton("Start")
# startButton.pressed.connect(buttonPressed)
outputLabel = QLabel("label")
# outputLabel.show()
mainLayout.addWidget(startButton)
mainLayout.addWidget(outputLabel)
widget.setLayout(mainLayout)
thread = MyThread()

def updateLabel(num):
	outputLabel.setText(str(num))

thread.updateProgress.connect(updateLabel)
startButton.pressed.connect(thread.start)
# QMainWindow using QWidget as central widget
# window = QMainWindow(widget)
widget.show()
#window.show()
# widget.generate()
app.exec_()