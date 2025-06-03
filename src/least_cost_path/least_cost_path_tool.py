from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingOutputVectorLayer,
    QgsProcessing
)
from src.least_cost_path.least_cost_path import least_cost_path_analysis

class LeastCostPathAnalysisAlgorithm(QgsProcessingAlgorithm):
    """
    Алгоритм построения путей наименьшей стоимости между точками с учетом рельефа и водных преград
    """

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return LeastCostPathAnalysisAlgorithm()

    def name(self):
        return 'leastcostpathanalysis'

    def displayName(self):
        return self.tr('Построение путей наименьшей стоимости')

    def group(self):
        return self.tr('Гидрология')

    def groupId(self):
        return 'hydrology'

    def shortHelpString(self):
        return self.tr(
            "Построение путей наименьшей стоимости между точками с учетом рельефа и водных преград.\n"
        )

    def initAlgorithm(self, config=None):
        # Входные параметры
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                'POINTS_LAYER',
                self.tr('Точечный слой с источниками'),
                types=[QgsProcessing.TypeVectorPoint]
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'DEM_LAYER',
                self.tr('Растр высот рельефа(DEM Layer)')
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                'WATER_RASTER_LAYER',
                self.tr('Растр водных объектов')
            )
        )

        # Выходные данные
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'LCP_LAYER',
                self.tr('Слой путей наименьшей стоимости')
            )
        )

        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'MOVED_SOURCES_LAYER',
                self.tr('Слой перемещенных источников')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Получаем входные параметры
        points_layer = self.parameterAsVectorLayer(parameters, 'POINTS_LAYER', context)
        dem_layer = self.parameterAsRasterLayer(parameters, 'DEM_LAYER', context)
        water_raster_layer = self.parameterAsRasterLayer(parameters, 'WATER_RASTER_LAYER', context)

        # Проверка наличия поля 'z' в точечном слое
        if 'z' not in [field.name() for field in points_layer.fields()]:
            feedback.reportError(self.tr("Ошибка: Точечный слой должен содержать поле 'z' с высотами точек"), True)
            return {}

        feedback.pushInfo(self.tr("Начало анализа путей наименьшей стоимости..."))

        # Вызываем основную функцию        
        try:
            lcp_layer, moved_sources_layer = least_cost_path_analysis(
                points_layer,
                dem_layer,
                water_raster_layer,
                feedback
            )
        except ValueError as e:
            error_msg = self.tr("Ошибка при выполнении анализа: ") + str(e)
            feedback.reportError(error_msg, True)
            return {}
        except Exception as e:
            error_msg = self.tr("Неожиданная ошибка: ") + str(e)
            feedback.reportError(error_msg, True)
            raise

        feedback.pushInfo(self.tr("Анализ успешно завершен!"))
        
        # Возвращаем результаты
        return {
            'LCP_LAYER': lcp_layer,
            'MOVED_SOURCES_LAYER': moved_sources_layer
        }