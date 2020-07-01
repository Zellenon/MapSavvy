# Original Credit to John Olsson for the math and code on the Voss Fractal Faulting Algorithm project
# https://www.lysator.liu.se/~johol/fwmg/howisitdone.html

# Credit to Wang Yasai for his Perlin Noise projects

import json
import random, sys, math, time
import numpy as np
import numpy.random as nprand
np.set_printoptions(threshold=np.inf, linewidth=10000)
from PySide2.QtWidgets import *
from MapGUI import *
from PathManager import PathManager

if __name__ == "__main__":
	PathManager()
	# Qt Application
	app = QApplication(sys.argv)
	# QWidget
	widget = Widget()
	# QMainWindow using QWidget as central widget
	window = MainWindow(widget)
	window.show()
	widget.generate()

	sys.exit(app.exec_())
