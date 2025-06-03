from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterVectorDestination,
    QgsProcessing
)
from .layers.polygon_from_path import flood_fill_areas, process_flood_fill

class CreateRastrAreasAlgorithm(QgsProcessingAlgorithm):
    """
    Интерактивный инструмент для заливки областей
    """
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_VECTOR = 'INPUT_VECTOR'
    MIN_AREA = 'MIN_AREA'  
    OUTPUT_RASTER = 'OUTPUT_RASTER'


    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CreateRastrAreasAlgorithm()

    def name(self):
        return 'flood_fill_areas'

    def displayName(self):
        return self.tr('Растр водоразделов')

    def group(self):
        return self.tr('Гидрология')

    def groupId(self):
        return 'hydrology'

    def shortHelpString(self):
        return self.tr("Алгоритм создаст растровый слой, ограниченный путями")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_RASTER,
            self.tr('Водный растр')
        ))
        
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_VECTOR,
            self.tr('Слой путей (линии)'),
            [QgsProcessing.TypeVectorLine]
        ))
        
        
        
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_RASTER,
            self.tr('Временный растр водоразделов')
        ))
        
        

    def processAlgorithm(self, parameters, context, feedback):
        # Получаем параметры
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        vector_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)

        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        
        # Создаём растровый слой водораздела
        try:
            flood_fill_areas(
                raster_layer.source(),
                vector_layer.source(),
                output_raster,
                feedback
            )
        except Exception as e:
            feedback.reportError(self.tr("Ошибка при обработке растра: ") + str(e), True)
            return {}
        
        
        
        return {self.OUTPUT_RASTER: output_raster}
    
class FloodFillPostProcessing(QgsProcessingAlgorithm):
    """
    Алгоритм для постобработки растрового слоя водоразделов
    """
    
    INPUT_RASTER = 'INPUT_RASTER'
    MIN_AREA = 'MIN_AREA'
    OUTPUT_VECTOR = 'OUTPUT_VECTOR'
    
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)
    
    def createInstance(self):
        return FloodFillPostProcessing()
    
    def name(self):
        return 'floodfillpostprocessing'
    
    def displayName(self):
        return self.tr('Полигоны из растра водоразделов')
    
    def group(self):
        return self.tr('Гидрология')
    
    def groupId(self):
        return 'hydrology'
    
    def shortHelpString(self):
        return self.tr(
            "Алгоритм выполняет постобработку растра водоразделов с заливанием областей, "
            "преобразуя его в полигоны и фильтруя по минимальной площади."
        )
    
    def initAlgorithm(self, config=None):
        # Входной растр
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Входной растр')
            )
        )
        
        # Минимальная площадь
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_AREA,
                self.tr('Минимальная площадь (кв.м)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=496665.0,
                minValue=0.0
            )
        )
        
        # Выходной вектор
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR,
                self.tr('Выходной векторный слой')
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        # Получаем параметры
        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        min_area = self.parameterAsDouble(parameters, self.MIN_AREA, context)
        output_vector = self.parameterAsOutputLayer(parameters, self.OUTPUT_VECTOR, context)
        
        # Вызываем нашу функцию        
        try:
            result = process_flood_fill(
                input_raster.source(),
                output_vector,
                min_area,
                feedback
            )
        except Exception as e:
            feedback.reportError(self.tr("Ошибка при создании полигонов водораздела растра: ") + str(e), True)
            return {}
        
        # Возвращаем результат
        return {self.OUTPUT_VECTOR: result}