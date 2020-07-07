import json
import random, sys, math, time
import numpy as np
import numpy.random as nprand
from PySide2.QtCore import QObject, QThread, Signal, Slot
np.set_printoptions(threshold=np.inf, linewidth=10000)

from PySide2.QtCore import Qt, Slot, QSize
from PySide2.QtGui import QPixmap, QImage, qRgb
from PySide2.QtWidgets import *
from PathManager import PathManager

def genSeed():
	words = [w.strip() for w in nprand.choice(open("data/corncob_lowercase.txt").readlines(),4)]
	words = [{0:w.upper(),1:w.lower()}[random.randint(0,1)] for w in words]
	return " ".join(words)

def seed_NameToNum(s):
	return int("".join([str(ord(w)) for w in s])) % (2**32)

class tracker:
	def __init__(self):
		self.timer = time.time()

	def tick(self):
		x = time.time() - self.timer
		self.timer = time.time()
		return x

class MapGenerator(QThread):
	displayCompleted = Signal(int)
	updateProgress = Signal(float)
	printUpdate = Signal(str)

	# User-set Variables:
	seedname = "abdc"
	faultCount = 0
	percentWater = 60
	shift = 0
	scale = 1
	xRange = -1
	yRange = -1
	xRangeRange = range(0, 1)

	image = QImage(50, 50, QImage.Format_Indexed8)

	# System Variables
	WorldMapArray = []  # 2D height map
	SinIterPhi = []  # Helps map 2D values to 3D
	Red = [0, 0, 0, 0, 0, 0, 0, 0, 34, 68, 102, 119, 136, 153, 170, 187,
	       0, 34, 34, 119, 187, 255, 238, 221, 204, 187, 170, 153,
	       136, 119, 85, 68,
	       255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200,
	       195, 190, 185, 180, 175]
	Green = [0, 0, 17, 51, 85, 119, 153, 204, 221, 238, 255, 255, 255,
	         255, 255, 255, 68, 102, 136, 170, 221, 187, 170, 136,
	         136, 102, 85, 85, 68, 51, 51, 34,
	         255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200,
	         195, 190, 185, 180, 175]
	Blue = [0, 68, 102, 136, 170, 187, 221, 255, 255, 255, 255, 255,
	        255, 255, 255, 255, 0, 0, 0, 0, 0, 34, 34, 34, 34, 34, 34,
	        34, 34, 34, 17, 0,
	        255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200,
	        195, 190, 185, 180, 175]
	yRangeDiv2 = 0.0
	yRangeDivPi = 0.0
	index2 = 0

	def __init__(self):
		QThread.__init__(self)

	def configure(self, seed= -1, faultNum= -1, percentWater= 60, size= (1000,500)):
		# Seed Generation or Interpretation
		if type(seed) == str:
			self.seedname = seed
		else:
			self.seedname = genSeed()
		self.seed = seed_NameToNum(self.seedname)

		# Deciding Number of Faults
		self.faultNum = faultNum
		if self.faultNum < 0:
			self.faultNum = random.randint(10,70) * 1000
		
		# Deciding Water Coverage %
		self.percentWater = percentWater
		if not 100 >= self.percentWater >= 0:
			self.percentWater = 60
		
		# Size of The Map (in pixels)
		if type(size) == tuple and len(size)==2:
			self.xRange = size[0]
			self.yRange = size[1]
		else:
			self.xRange = 1000
			self.yRange = 500

	def run(self):
		self.updateProgress.emit(0)
		nprand.seed(self.seed)
		self.printUpdate.emit(self.seedname)
		self.printUpdate.emit(self.seed)

		self.image = QImage(self.xRange, self.yRange, QImage.Format_Indexed8)
		[self.image.setColor(w, qRgb(self.Red[w], self.Green[w], self.Blue[w])) for w in range(49)]

		self.WorldMapArray = np.zeros((self.yRange, self.xRange))
		self.SinIterPhi = np.array(range(self.xRange * 2))
		self.SinIterPhi = self.SinIterPhi.astype(float)
		self.SinIterPhi[:len(self.SinIterPhi) // 2] = np.sin(self.SinIterPhi[:len(self.SinIterPhi) // 2] * 2 * math.pi / self.xRange)
		self.SinIterPhi[len(self.SinIterPhi) // 2:] = self.SinIterPhi[:len(self.SinIterPhi) // 2]

		self.xRangeRange = range(0, self.xRange)
		self.yRangeDiv2 = self.yRange / 2.
		self.yRangeDivPi = self.yRange / math.pi

		self.faultCount = 0
		while (self.faultCount < self.faultNum):
			self.AddFaults(self.faultNum) #should be //5
			self.updateProgress.emit(100)
			self.display(0)
			self.updateProgress.emit(200)
		#self.display(0)
		self.printUpdate.emit("done")

	def AddFaults(self, newFaultNum):
		rawNumbers = nprand.random(size=newFaultNum * 4)
		rawNumbers.resize(newFaultNum,4)
		self.flag = np.array([-1,1])[rawNumbers.round().astype(int)[:,0]]
		self.alpha = (rawNumbers[:,1]-0.5)*np.pi
		self.beta = (rawNumbers[:,2]-0.5)*np.pi
		self.shift = rawNumbers[:,3] * self.xRange

		self.tanB = np.tan(np.arccos(np.cos(self.alpha) * np.cos(self.beta)))
		self.xsi = (self.xRange / 2 - (self.xRange / math.pi) * self.beta)

		for i in range(newFaultNum):
			if (self.faultCount % (self.faultNum//10) == 0):
				self.printUpdate.emit("Fault " + str(i - 1) + " (" + str(self.faultCount * 100 // self.faultNum) + "%)")
				self.updateProgress.emit(self.faultCount * 100 // self.faultNum)
			self.GenerateFault(i)
			self.faultCount += 1

	def GenerateFault(self, fi: int): # fi = fault index
		#                     Runs once for each fault
		SinIterIndex = np.array(self.xRangeRange)
		SinIterIndex = (self.xsi[fi] - SinIterIndex+self.xRange).astype(int)
		SinIterIndex[SinIterIndex > len(self.SinIterPhi)-1] -= 1
		SinIterIndexes = ((SinIterIndex+self.shift[fi]) % len(self.SinIterPhi)-1).astype(int)
		atanArgs = self.SinIterPhi[SinIterIndexes] * self.tanB[fi]
		theta = (self.yRangeDivPi * np.arctan(atanArgs) + self.yRangeDiv2).astype(int)
		theta[theta > self.yRange - 1] -= 1

		self.WorldMapArray[theta,self.xRangeRange] +=self.flag[fi]

	def display(self, displayMode: int): #0 for regular, 1 for fault debug, 2 for height
		DISPLAYTIMER = tracker()

		if (displayMode == 1):
			displayMap = (self.WorldMapArray - self.WorldMapArray.min()) / (self.WorldMapArray.max() - self.WorldMapArray.min()) * 48
			self.updateProgress.emit(120)
			[[self.image.setPixel(x, y, value) for x, value in enumerate(ylist)] for y, ylist in enumerate(displayMap)]

			self.printUpdate.emit("Fault Debugging: " + str(DISPLAYTIMER.tick()))
		elif displayMode == 0 or displayMode == 2:
			self.ColorMap = np.zeros((self.yRange,self.xRange), dtype=float)
			Histogram = np.zeros(32, dtype=int)
			Color = 0
			for x in self.xRangeRange:
				self.ColorMap[:,x] = self.WorldMapArray[:,x].cumsum()

			self.printUpdate.emit("Color Mapping: " + str(DISPLAYTIMER.tick()))
			self.updateProgress.emit(125)

			minimum = self.ColorMap.min()
			maximum = self.ColorMap.max()

			HistoCalcArray = np.copy(self.ColorMap)
			HistoCalcArray = HistoCalcArray.astype(int)
			HistoCalcArray = ((HistoCalcArray - HistoCalcArray.min()) / (HistoCalcArray.max() - HistoCalcArray.min())) * 30 + 1
			HistoCalcArray = HistoCalcArray.astype(int)
			x, Histogram = np.unique(HistoCalcArray,return_counts=True)

			self.printUpdate.emit("Histogram Calculation: " + str(DISPLAYTIMER.tick()))
			self.updateProgress.emit(150)

			Threshold = (self.percentWater * self.xRange * self.yRange) / 100
			Count = 0
			j = 0
			for j in range(len(Histogram)):
				Count += Histogram[j]
				if (Count > Threshold):
					break
			Threshold = j * (maximum - minimum + 1) / 30 + minimum

			# Scale ColorMap to colorrange in a way that gives you
			# a certain Ocean / Land ratio
			shape = self.ColorMap.shape
			self.ColorMap = self.ColorMap.reshape(shape[0]*shape[1])
			colorCopy = self.ColorMap.copy()
			self.ColorMap[self.ColorMap<Threshold] = (colorCopy[colorCopy<Threshold] - minimum) / (Threshold - minimum) * 15. #+1.
			self.ColorMap[colorCopy >= Threshold] = (colorCopy[colorCopy >= Threshold] - Threshold) / (maximum - Threshold) * 15. + 16.
			self.ColorMap = self.ColorMap.astype(int)
			self.ColorMap = self.ColorMap.reshape(shape)

			self.printUpdate.emit("Colormap Weighting: " + str(DISPLAYTIMER.tick()))
			self.updateProgress.emit(175)

			self.printUpdate.emit(self.ColorMap.shape)
			if (displayMode == 2):
				for y,ylist in enumerate(self.ColorMap):
					for x, value in enumerate(ylist):
						self.image.setPixel(x, y, value)
			else:
				[[self.image.setPixel(x, y, value) for x, value in enumerate(ylist)] for y, ylist in enumerate(self.ColorMap)]
			#self.image = QImage(self.ColorMap,self.xRange,self.yRange)

			if displayMode == 0:
				[self.image.setColor(w, qRgb(self.Red[w], self.Green[w], self.Blue[w])) for w in range(49)]
			else:
				greyScaleValues = [255*w//49 for w in range(49)]
				[self.image.setColor(w, qRgb(greyScaleValues[w], greyScaleValues[w], greyScaleValues[w])) for w in range(49)]
			self.printUpdate.emit("Image Setting: " + str(DISPLAYTIMER.tick()))
		self.displayCompleted.emit(displayMode)

class MapGenerator2(QObject):
	displayCompleted = Signal(int)

	# User-set Variables:
	seedname = "abdc"
	faultCount = 0

	percentWater = 60
	shift = 0
	scale = 1
	xRange = -1
	yRange = -1
	xRangeRange = range(0, 1)

	image = QImage(50, 50, QImage.Format_Indexed8)

	# System Variables
	WorldMapArray = []  # 2D height map
	SinIterPhi = []  # Helps map 2D values to 3D
	Red = [0, 0, 0, 0, 0, 0, 0, 0, 34, 68, 102, 119, 136, 153, 170, 187,
	       0, 34, 34, 119, 187, 255, 238, 221, 204, 187, 170, 153,
	       136, 119, 85, 68,
	       255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200,
	       195, 190, 185, 180, 175]
	Green = [0, 0, 17, 51, 85, 119, 153, 204, 221, 238, 255, 255, 255,
	         255, 255, 255, 68, 102, 136, 170, 221, 187, 170, 136,
	         136, 102, 85, 85, 68, 51, 51, 34,
	         255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200,
	         195, 190, 185, 180, 175]
	Blue = [0, 68, 102, 136, 170, 187, 221, 255, 255, 255, 255, 255,
	        255, 255, 255, 255, 0, 0, 0, 0, 0, 34, 34, 34, 34, 34, 34,
	        34, 34, 34, 17, 0,
	        255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200,
	        195, 190, 185, 180, 175]
	yRangeDiv2 = 0.0
	yRangeDivPi = 0.0
	index2 = 0

	# --------------------------------------------------
	# ---------------- GENERATION CODE -----------------
	# --------------------------------------------------
	def GenerateNewMap(self, seed= -1, faultNum= -1, percentWater= 60, size= (1000,500)):
		# Seed Generation or Interpretation
		if type(seed) == str:
			self.seedname = seed
		else:
			self.seedname = genSeed()
		self.seed = seed_NameToNum(self.seedname)

		# Deciding Number of Faults
		if faultNum > 0:
			faultNum = faultNum #									 CHANGE THIS LATER
		else:
			#self.faultNum = int(10 * (pow(10, 2.2+2*random())))
			faultNum = random.randint(10,70) * 100
			#self.faultNum = 2000
		
		# Deciding Water Coverage %
		self.percentWater = percentWater
		if not 100 >= self.percentWater >= 0:
			self.percentWater = 60
		
		# Size of The Map (in pixels)
		if type(size) == tuple and len(size)==2:
			self.xRange = size[0]
			self.yRange = size[1]
		else:
			self.xRange = 1000
			self.yRange = 500

		nprand.seed(self.seed)
		print(self.seedname)
		print(self.seed)

		self.image = QImage(self.xRange, self.yRange, QImage.Format_Indexed8)
		[self.image.setColor(w, qRgb(self.Red[w], self.Green[w], self.Blue[w])) for w in range(49)]

		self.WorldMapArray = np.zeros((self.yRange, self.xRange))
		self.SinIterPhi = np.array(range(self.xRange * 2))
		self.SinIterPhi = self.SinIterPhi.astype(float)
		self.SinIterPhi[:len(self.SinIterPhi) // 2] = np.sin(self.SinIterPhi[:len(self.SinIterPhi) // 2] * 2 * math.pi / self.xRange)
		self.SinIterPhi[len(self.SinIterPhi) // 2:] = self.SinIterPhi[:len(self.SinIterPhi) // 2]

		self.xRangeRange = range(0, self.xRange)
		self.yRangeDiv2 = self.yRange / 2.
		self.yRangeDivPi = self.yRange / math.pi

		self.faultCount = 0
		while (self.faultCount < faultNum):
			self.AddFaults(faultNum) #should be //5
			self.display(0)

		print("displaying")
		#self.display(0)
		print("done")

	def AddFaults(self, faultNum):
		rawNumbers = nprand.random(size=faultNum * 4)
		rawNumbers.resize(faultNum,4)
		#self.flag = nprand.choice((-1, 1), size=faultNum)
		self.flag = np.array([-1,1])[rawNumbers.round().astype(int)[:,0]]
		#self.alpha = nprand.uniform(low=-0.5, high=0.5, size=faultNum) * math.pi
		self.alpha = (rawNumbers[:,1]-0.5)*np.pi
		#self.beta = nprand.uniform(low=-0.5, high=0.5, size=faultNum) * math.pi
		self.beta = (rawNumbers[:,2]-0.5)*np.pi
		#self.shift = nprand.random(size=faultNum) * self.xRange
		self.shift = rawNumbers[:,3] * self.xRange

		self.tanB = np.tan(np.arccos(np.cos(self.alpha) * np.cos(self.beta)))
		self.xsi = (self.xRange / 2 - (self.xRange / math.pi) * self.beta)

		milestones = list(range(1, faultNum, faultNum // 10))
		for i in range(faultNum):
			if (i in milestones):
				print("Fault " + str(i - 1) + " (" + str(i * 100 // faultNum) + "%)")
			self.GenerateFault(i)

		self.faultCount += faultNum

	def GenerateFault(self, fi: int): # fi = fault index
		#                     Runs once for each fault
		SinIterIndex = np.array(self.xRangeRange)
		SinIterIndex = (self.xsi[fi] - SinIterIndex+self.xRange).astype(int)
		SinIterIndex[SinIterIndex > len(self.SinIterPhi)-1] -= 1
		SinIterIndexes = ((SinIterIndex+self.shift[fi]) % len(self.SinIterPhi)-1).astype(int)
		atanArgs = self.SinIterPhi[SinIterIndexes] * self.tanB[fi]
		theta = (self.yRangeDivPi * np.arctan(atanArgs) + self.yRangeDiv2).astype(int)
		theta[theta > self.yRange - 1] -= 1

		self.WorldMapArray[theta,self.xRangeRange] +=self.flag[fi]
		#subsection = range(1000, 1500)
		#self.WorldMapArray[theta[1000:1500], subsection] += self.flag[fi]


	def FloodFill4(self, x: int, y: int, OldColor: int): # '4-connective floodfill algorithm'. Used for the ice-caps.
		pass

	# --------------------------------------------------
	# ---------------- DISPLAY CODE --------------------
	# --------------------------------------------------
	def display(self, displayMode: int): #0 for regular, 1 for fault debug
		DISPLAYTIMER = tracker()

		if (displayMode == 1):
			displayMap = (self.WorldMapArray - self.WorldMapArray.min()) / (self.WorldMapArray.max() - self.WorldMapArray.min()) * 48
			[[self.image.setPixel(x, y, value) for x, value in enumerate(ylist)] for y, ylist in enumerate(displayMap)]

			print("Fault Debugging: " + str(DISPLAYTIMER.tick()))
		else:

			self.ColorMap = np.zeros((self.yRange,self.xRange), dtype=float)
			Histogram = np.zeros(32, dtype=int)
			Color = 0
			'''for x in range(self.xRange):
				Color = self.WorldMapArray[0][x]
				for i in range(self.yRange):
					Color += self.WorldMapArray[i][x]
					self.ColorMap[i][x] = Color'''
			for x in self.xRangeRange:
				self.ColorMap[:,x] = self.WorldMapArray[:,x].cumsum()


			print("Color Mapping: " + str(DISPLAYTIMER.tick()))

			minimum = self.ColorMap.min()
			maximum = self.ColorMap.max()

			HistoCalcArray = np.copy(self.ColorMap)
			HistoCalcArray = HistoCalcArray.astype(int)
			HistoCalcArray = ((HistoCalcArray - HistoCalcArray.min()) / (HistoCalcArray.max() - HistoCalcArray.min())) * 30 + 1
			HistoCalcArray = HistoCalcArray.astype(int)
			x, Histogram = np.unique(HistoCalcArray,return_counts=True)

			print("Histogram Calculation: " + str(DISPLAYTIMER.tick()))

			Threshold = (self.percentWater * self.xRange * self.yRange) / 100
			Count = 0
			j = 0
			for j in range(len(Histogram)):
				Count += Histogram[j]
				if (Count > Threshold):
					break
			Threshold = j * (maximum - minimum + 1) / 30 + minimum

			# Scale ColorMap to colorrange in a way that gives you
			# a certain Ocean / Land ratio
			shape = self.ColorMap.shape
			self.ColorMap = self.ColorMap.reshape(shape[0]*shape[1])
			colorCopy = self.ColorMap.copy()
			self.ColorMap[self.ColorMap<Threshold] = (colorCopy[colorCopy<Threshold] - minimum) / (Threshold - minimum) * 15. #+1.
			self.ColorMap[colorCopy >= Threshold] = (colorCopy[colorCopy >= Threshold] - Threshold) / (maximum - Threshold) * 15. + 16.
			self.ColorMap = self.ColorMap.astype(int)
			self.ColorMap = self.ColorMap.reshape(shape)

			print("Colormap Weighting: " + str(DISPLAYTIMER.tick()))

			''' PEAK CULLING PROTOTYPE
			* p = perlin noise
			* c = most extreme possible reduction(probably 0.6x?)
			*pc = local reduction
			* l = local level
			* l = -pc(l - 1) ^ 2 + pc < - Only works on values 0 - 1. Map somehow.
			* l = M * (-1 * c * ((x - m) / M - 1) ^ 2 + c) + m < - Mapped version((M) ax, (m) in )
			* // *min = 0; max = 0;
			for (int i = 0; i < xRange * yRange; i++) {
				min = min(min, ColorMap[i / xRange][i % xRange]);
				max = max(max, ColorMap[i / xRange][i % xRange]);
			}
			float c = 0.4, pc = 0;
			for (int x = 0; x < xRange; x++) {
				for (int y = 0; y < yRange; y++) {
					int l = ColorMap[y][x];
					pc = noise(x * 0.01, y * 0.01) * 0.5+0.5; // Whatever is after noise() is used to map the noise.Noise() can
						return 0, and we don 't want that, so we have to change it SOMEhow.
					float mix = noise(x * 0.01 + 1, y * 0.01);
					l = (int)((max * (-1 * pc * pow(((float(l) - min) / max - 1), 2) + pc) + min) * mix + ((1 - mix) * l));
					// l = (int)(pc * 48); ColorMap[y][x] = l; 
				} 
			}'''
			print(self.ColorMap.shape)
			if (displayMode == 2):
				for y,ylist in enumerate(self.ColorMap):
					for x, value in enumerate(ylist):
						self.image.setPixel(x, y, value)
			else:
				[[self.image.setPixel(x, y, value) for x, value in enumerate(ylist)] for y, ylist in enumerate(self.ColorMap)]
			#self.image = QImage(self.ColorMap,self.xRange,self.yRange)

			print("Image Setting: " + str(DISPLAYTIMER.tick()))
		self.displayCompleted.emit(displayMode)
