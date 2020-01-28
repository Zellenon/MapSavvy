import json
import random, sys, math, time
import numpy as np
import numpy.random as nprand
np.set_printoptions(threshold=np.inf, linewidth=10000)

from PySide2.QtCore import Qt, Slot, QSize
from PySide2.QtGui import QPixmap, QImage, qRgb
from PySide2.QtWidgets import *
from PathManager import PathManager

def genSeed():
	words = [w.strip() for w in nprand.choice(open("../data/corncob_lowercase.txt").readlines(),4)]
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

class MapGenerator:
	# User - set Variables:
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

	def startup(self):
		if self.seedname == -1:
			self.seedname = ''.join([chr(w) for w in nprand.randint(ord('a'),ord('z'),7)])
		self.seed = sum( [ord(v)*(9**(w+1)) for w,v in enumerate(self.seedname)] )
		print(self.seedname)
		print(self.seed)
		nprand.seed(self.seed)
		self.image = QImage(self.xRange, self.yRange, QImage.Format_Indexed8)
		[self.image.setColor(w, qRgb(self.Red[w], self.Green[w], self.Blue[w])) for w in range(49)]

		# Beginning of 'Main'


	# --------------------------------------------------
	# ---------------- GENERATION CODE -----------------
	# --------------------------------------------------
	def GenerateNewMap(self, **kwargs):
		if 'faultNum' in kwargs and kwargs['faultNum']>0:
			faultNum = kwargs['faultNum']
		else:
			#self.faultNum = int(10 * (pow(10, 2.2+2*random())))
			faultNum = random.randint(20,70) * 1000
			#self.faultNum = 2000
		if 'percentWater' in kwargs:
			self.percentWater = kwargs['percentWater']
		if not 100 >= self.percentWater >= 0:
			self.percentWater = 60
		if ('size' in kwargs and len(kwargs)==2):
			self.xRange = kwargs['size'][0]
			self.yRange = kwargs['size'][1]
		else:
			self.xRange = 1000
			self.yRange = 500
		if ('seed' in kwargs and type(kwargs['seed']) == str):
			self.seedname = kwargs['seed']
		else:
			self.seedname = genSeed()
		self.seed = seed_NameToNum(self.seedname)
		nprand.seed(self.seed)
		print(self.seedname)
		print(self.seed)

		self.WorldMapArray = np.zeros((self.yRange, self.xRange))
		self.image = QImage(self.xRange, self.yRange, QImage.Format_Indexed8)
		[self.image.setColor(w, qRgb(self.Red[w], self.Green[w], self.Blue[w])) for w in range(49)]
		self.SinIterPhi = np.array(range(self.xRange * 2))
		self.SinIterPhi = self.SinIterPhi.astype(float)
		self.SinIterPhi[:len(self.SinIterPhi) // 2] = np.sin(self.SinIterPhi[:len(self.SinIterPhi) // 2] * 2 * math.pi / self.xRange)
		self.SinIterPhi[len(self.SinIterPhi) // 2:] = self.SinIterPhi[:len(self.SinIterPhi) // 2]

		self.xRangeRange = range(0, self.xRange)
		self.yRangeDiv2 = self.yRange / 2.
		self.yRangeDivPi = self.yRange / math.pi

		self.faultCount = 0
		self.AddFaults(faultNum)

		print("displaying")
		self.display(0)
		print("done")

	def AddFaults(self, faultNum):
		self.flag = nprand.choice((-1, 1), size=faultNum)
		self.alpha = nprand.uniform(low=-0.5, high=0.5, size=faultNum) * math.pi
		self.beta = nprand.uniform(low=-0.5, high=0.5, size=faultNum) * math.pi
		self.shift = nprand.random(size=faultNum) * self.xRange

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
			return

		self.ColorMap = np.zeros((self.yRange,self.xRange), dtype=float) # THIS LINE CAUSES THE STRIPES. IT __SHOULD__ BE OVERWRITTEN!
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
			[[self.image.setPixel(x, y, value) for x, value in enumerate(ylist)] for y, ylist in enumerate(self.ColorMap)]
		[[self.image.setPixel(x,y,value) for x,value in enumerate(ylist)] for y,ylist in enumerate(self.ColorMap)]
		#self.image = QImage(self.ColorMap,self.xRange,self.yRange)

		print("Image Setting: " + str(DISPLAYTIMER.tick()))