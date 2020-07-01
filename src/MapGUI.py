from random import *
from PathManager import PathManager
from MapGenerator import *
from PySide2.QtCore import Qt, Slot, QSize, Signal, QObject
from PySide2.QtGui import QPixmap, QImage, qRgb
from PySide2.QtWidgets import *

class Widget(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.startup()

    def startup(self):
        self.generator = MapGenerator()

    def generate(self):
        self.layout = QHBoxLayout()
        self.display = QLabel()
        self.generator.displayCompleted.connect(self.updateImage)
        #self.display.setPixmap(QPixmap.fromImage(self.generator.image))
        self.layout.addWidget(self.display)
        self.setLayout(self.layout)

        scale = 0.5
        size = (int(3000 * scale), int(2000 * scale))
        self.generator.GenerateNewMap(size=size)

    def updateImage(self):
        self.display.setPixmap(QPixmap.fromImage(self.generator.image))

    def save(self):
        print(self.generator.image.save(PathManager.path("images") + self.generator.seedname + " - " + str(self.generator.faultCount) + ".png"))

class MainWindow(QMainWindow):

    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.widget = widget
        #geometry = app.desktop().availableGeometry(self)
        #self.setFixedSize(geometry.width() * 0.95, geometry.height() * 0.90)
        self.setFixedSize(3000,2000)
        self.setWindowTitle("MapSavvy")

        PathManager.insureFolder(PathManager.libraryPath+"images")

        widget.setFixedSize(self.size())

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        # Saving the Image
        save_action = QAction("Save", self)
        #save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save)

        self.file_menu.addAction(save_action)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.status = self.statusBar()
        self.status.showMessage("Loaded and plotted")

        self.setCentralWidget(widget)

    def save(self):
        self.widget.save()

    @Slot()
    def exit_app(self, checked):
        sys.exit()
