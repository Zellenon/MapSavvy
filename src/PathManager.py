import PySide2, json, os, random, math, operator, cv2
from PySide2.QtGui import QMovie
from PySide2.QtCore import *

class PathManager:
	@classmethod
	def __init__(cls):
		cls.localConfigFolder = QStandardPaths.writableLocation(QStandardPaths.DataLocation) + "/MapSavvy/"
		cls.localConfigFile = cls.localConfigFolder + "paths.cfg"
		cls.insureFolder(cls.localConfigFolder)
		if (not os.path.isfile(cls.localConfigFile)):
			file = open(cls.localConfigFile, 'w')
			file.write(cls.localConfigFolder + "data/")
			file.close()

		file = open(cls.localConfigFile, 'r')
		cls.libraryPath = file.read()
		cls.libraryPath = cls.libraryPath.replace("\n","")
		cls.insureFolder(cls.libraryPath)
		print(cls.libraryPath)

	@classmethod
	def insureFolder(cls, path):
		if not os.path.isdir(path):
			os.mkdir(path)

	@classmethod
	def path(cls, path):
		return cls.libraryPath + path + "/"