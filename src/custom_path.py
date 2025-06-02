from pathlib import Path

import networkit as nk
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsLineSymbol,
    QgsPointXY,
    QgsProject,  # Added QgsProject import
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import (
    QMessageBox,
    QPushButton,
)
from qgis.utils import iface

from .least_cost_path.least_cost_path import (
    build_cost_graph,
    calculate_minimum_elevation,
    coord_to_pixel,
    pixel_to_coord,
)
from .river.point_selection_tool import PointSelectionTool


class CustomPathBuilder:
    def __init__(self, project_folder: Path) -> None:
        self.project_folder = project_folder
        self.custom_path_button = None
        self.point_tool = None

    def add_custom_path_button(self, iface) -> None:
        """Добавляет кнопку для построения пользовательского пути."""
        if self.custom_path_button:
            self.custom_path_button.deleteLater()

        self.custom_path_button = QPushButton("Построить путь между точками")
        self.custom_path_button.setFixedWidth(200)
        self.custom_path_button.clicked.connect(self.run_custom_path_selection)
        iface.addToolBarWidget(self.custom_path_button)

    def run_custom_path_selection(self) -> None:
        """Запускает инструмент выбора точек на карте."""
        canvas = iface.mapCanvas()
        self.point_tool = PointSelectionTool(canvas, points=2)
        self.point_tool.selection_completed.connect(self.process_custom_path)
        canvas.setMapTool(self.point_tool)

    def process_custom_path(self, points) -> None:
        """Обрабатывает выбранные точки и строит путь."""
        if len(points) != 2:
            QMessageBox.warning(None, "Ошибка", "Необходимо выбрать ровно 2 точки")
            return

        self.build_path_between_points(points[0], points[1])

    def cleanup(self) -> None:
        """Очищает ресурсы."""
        if self.custom_path_button:
            self.custom_path_button.deleteLater()
            self.custom_path_button = None

        if self.point_tool:
            canvas = iface.mapCanvas()
            if canvas.mapTool() == self.point_tool:
                canvas.unsetMapTool(self.point_tool)
            self.point_tool = None

    def build_path_between_points(self, start_point, end_point) -> None:
        """Строит путь наименьшей стоимости между двумя точками."""
        try:
            # Проверяем наличие DEM
            dem_path = Path(self.project_folder) / "srtm_output_3857_pooled.tif"
            if not dem_path.exists():
                dem_path = Path(self.project_folder) / "river_dem_3857.tif"
                if not dem_path.exists():
                    QMessageBox.warning(
                        None,
                        "Ошибка",
                        "DEM слой не найден. Сначала выполните анализ оптимальных путей.",
                    )
                    return

            water_rasterized = Path(self.project_folder) / "water_rasterized.tif"
            if not water_rasterized.exists():
                QMessageBox.warning(
                    None, "Ошибка", "Растеризованный слой воды не найден."
                )
                return

            # Загружаем DEM
            dem_layer = QgsRasterLayer(str(dem_path), "DEM")
            if not dem_layer.isValid():
                QMessageBox.warning(None, "Ошибка", "Не удалось загрузить DEM слой")
                return

            # Строим граф стоимости с учетом нового модуля least_cost_path
            g, gt, n_rows, n_cols = build_cost_graph(dem_path, water_rasterized)

            # Преобразуем координаты
            src_crs = QgsProject.instance().crs()
            tgt_crs = QgsCoordinateReferenceSystem("EPSG:3857")
            transform = QgsCoordinateTransform(src_crs, tgt_crs, QgsProject.instance())

            start_3857 = transform.transform(start_point)
            end_3857 = transform.transform(end_point)

            # Конвертируем в пиксели
            start_i, start_j = coord_to_pixel(
                start_3857.x(), start_3857.y(), gt, n_rows, n_cols
            )
            end_i, end_j = coord_to_pixel(
                end_3857.x(), end_3857.y(), gt, n_rows, n_cols
            )

            # Проверяем, что точки находятся в пределах растра
            if not (0 <= start_i < n_rows and 0 <= start_j < n_cols):
                QMessageBox.warning(None, "Ошибка", "Начальная точка находится вне DEM")
                return

            if not (0 <= end_i < n_rows and 0 <= end_j < n_cols):
                QMessageBox.warning(None, "Ошибка", "Конечная точка находится вне DEM")
                return

            start_node = start_i * n_cols + start_j
            end_node = end_i * n_cols + end_j

            # Вычисляем путь
            dijk = nk.distance.Dijkstra(g, start_node)
            dijk.run()
            node_path = dijk.getPath(end_node)

            if not node_path:
                QMessageBox.information(
                    None, "Информация", "Путь между точками не найден"
                )
                return

            # Конвертируем узлы в координаты
            path_pts = [
                QgsPointXY(*pixel_to_coord(u // n_cols, u % n_cols, gt))
                for u in node_path
            ]

            # Создаем слой
            vl = QgsVectorLayer("LineString?crs=EPSG:3857", "Custom Path", "memory")
            pr = vl.dataProvider()
            pr.addAttributes([QgsField("id", QVariant.Int)])
            vl.updateFields()

            # Добавляем фичу
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(path_pts))
            feat.setAttributes([1])
            pr.addFeature(feat)

            # Настраиваем стиль
            symbol = QgsLineSymbol.createSimple(
                {"color": "255,0,0", "width": "2", "line_style": "solid"}
            )
            vl.setRenderer(QgsSingleSymbolRenderer(symbol))
            vl.triggerRepaint()

            # Добавляем слой
            QgsProject.instance().addMapLayer(vl)

            # Проверяем высоты
            min_elev = calculate_minimum_elevation(
                dem_layer, QgsGeometry.fromPolylineXY(path_pts)
            )
            if min_elev is not None:
                sample_start = dem_layer.dataProvider().sample(start_3857, 1)
                sample_end = dem_layer.dataProvider().sample(end_3857, 1)

                if sample_start and sample_end:
                    z_start, valid_start = sample_start
                    z_end, valid_end = sample_end
                    if valid_start and valid_end:
                        z1 = min(z_start, z_end)
                        if min_elev < z1 - 15:
                            QMessageBox.information(
                                None,
                                "Информация",
                                "Путь пересекает низменность (разница высот > 15м)",
                            )

            iface.mapCanvas().refresh()

        except Exception as e:
            QMessageBox.critical(None, "Ошибка", f"Не удалось построить путь: {e!s}")
