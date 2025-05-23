from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapToolEmitPoint, QgsVertexMarker


class PointSelectionTool(QgsMapToolEmitPoint):
    selection_completed = pyqtSignal(list)

    def __init__(self, canvas, points):
        super().__init__(canvas)
        self.points = []
        self.canvas = canvas
        self.markers = []
        self.required_points = points

    def canvasPressEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        self.points.append(point)

        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setColor(QColor(255, 0, 0))
        marker.setIconSize(10)
        self.markers.append(marker)

        if len(self.points) == self.required_points:
            self.selection_completed.emit(self.points)
            self.cleanup()

    def cleanup(self):
        for marker in self.markers:
            self.canvas.scene().removeItem(marker)
        self.markers = []
        self.canvas.unsetMapTool(self)
