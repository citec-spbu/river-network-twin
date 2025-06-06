from pathlib import Path
from typing import Optional

from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from qgis.utils import iface

from .custom_path import CustomPathBuilder
from .forest import forest
from .least_cost_path.least_cost_path import least_cost_path_analysis
from .river.river import river


class CustomDEMPlugin:
    def __init__(self, iface) -> None:
        self.iface = iface
        self.project_folder: Optional[Path] = None
        self.plugin_name = "RiverNETWORK"
        self.custom_path_builder = None

    def initGui(self) -> None:
        self.action = QAction(self.plugin_name, self.iface.mainWindow())
        self.action.triggered.connect(self.run_plugin)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&RiverNETWORK", self.action)

    def unload(self) -> None:
        """Удаление плагина."""
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&RiverNETWORK", self.action)
        if self.custom_path_builder:
            self.custom_path_builder.cleanup()
            self.custom_path_builder = None

    def run_plugin(self) -> None:
        # Код плагина
        folder = QFileDialog.getExistingDirectory(None, "Выберите рабочую папку")
        if not folder:
            QMessageBox.warning(
                None,
                "Ошибка",
                "Рабочая папка не выбрана. Работа плагина прекращена.",
            )
            return

        # Создать папку "work" внутри выбранной папки
        self.project_folder = Path(folder) / "work"
        self.project_folder.mkdir(exist_ok=True, parents=True)
        QMessageBox.information(
            None,
            "Папка установлена",
            f"Рабочая папка: {self.project_folder}",
        )
        self.run_programm()

    def show_layer_visibility_dialog(self) -> None:
        # Создать диалоговое окно
        dialog = QDialog()
        dialog.setWindowTitle("Выбор слоев")
        layout = QVBoxLayout()

        # Добавить лейбл
        label = QLabel("Выберите слои для отображения:")
        layout.addWidget(label)

        # Получить доступ к дереву слоев проекта
        project = QgsProject.instance()
        root = project.layerTreeRoot()

        # Создать чекбокс для каждого слоя
        checkboxes = {}
        for layer in project.mapLayers().values():
            layer_tree_node = root.findLayer(layer.id())
            if layer_tree_node:  # Проверка наличия слоя в дереве
                checkbox = QCheckBox(layer.name())
                checkbox.setChecked(
                    layer_tree_node.isVisible(),
                )  # Получить настоящую видимость
                layout.addWidget(checkbox)
                checkboxes[layer_tree_node] = checkbox

        # Добавить кнопку
        apply_button = QPushButton("Применить")
        layout.addWidget(apply_button)

        def apply_layer_visibility() -> None:
            for layer_tree_node, checkbox in checkboxes.items():
                layer_tree_node.setItemVisibilityChecked(checkbox.isChecked())
            dialog.close()

        # Соединить кнопку подтверждения с функцией
        apply_button.clicked.connect(apply_layer_visibility)

        # Настройка макета и отображение диалогового окна
        dialog.setLayout(layout)
        dialog.exec_()

    # Определите основную функцию для диалога
    def show_choice_dialog(self) -> None:
        # Создание диалогового окна
        dialog = QDialog()
        dialog.setWindowTitle("Выбор функции")
        layout = QVBoxLayout()

        # Добавление метки
        label = QLabel("Что вы хотите сделать?")
        layout.addWidget(label)

        # Добавляйте кнопки для различных вариантов
        waterlines_button = QPushButton("Создать речную сеть")
        waterlines_with_clustering_button = QPushButton(
            "Создать речную сеть с кластеризацией"
        )
        forest_belts_button = QPushButton("Создать лесополосы")
        cost_path_button = QPushButton("Вычислить путь наименьшей стоимости")
        cost_path_with_clustering_button = QPushButton(
            "Вычислить путь наименьшей стоимости с кластеризацией"
        )

        layout.addWidget(waterlines_button)
        layout.addWidget(waterlines_with_clustering_button)
        layout.addWidget(forest_belts_button)
        layout.addWidget(cost_path_button)
        layout.addWidget(cost_path_with_clustering_button)

        # Определение действий для кнопок
        def create_waterlines() -> None:
            dialog.close()
            river(self.project_folder, with_clustering=False)
            self.add_custom_path_button()

        def create_waterlines_with_clustering() -> None:
            dialog.close()
            river(self.project_folder, with_clustering=True)
            self.add_custom_path_button()

        def create_forest_belts() -> None:
            dialog.close()
            forest(self.project_folder)
            self.add_custom_path_button()

        def create_cost_path() -> None:
            dialog.close()
            river(self.project_folder, with_clustering=False)
            least_cost_path_analysis(self.project_folder)
            self.add_custom_path_button()

        def create_cost_path_with_clustering() -> None:
            dialog.close()
            river(self.project_folder, with_clustering=True)
            least_cost_path_analysis(self.project_folder)
            self.add_custom_path_button()

        # Свяжите кнопки с их действиями
        waterlines_button.clicked.connect(create_waterlines)
        waterlines_with_clustering_button.clicked.connect(
            create_waterlines_with_clustering
        )
        forest_belts_button.clicked.connect(create_forest_belts)
        cost_path_button.clicked.connect(create_cost_path)
        cost_path_with_clustering_button.clicked.connect(
            create_cost_path_with_clustering
        )

        # Настройка макета и отображение диалогового окна
        dialog.setLayout(layout)
        dialog.exec_()

    def add_custom_path_button(self) -> None:
        """Add the custom path builder button after algorithm completion."""
        if not self.custom_path_builder:
            self.custom_path_builder = CustomPathBuilder(self.project_folder)
        self.custom_path_builder.add_custom_path_button(self.iface)

    def clear_cache(self) -> None:
        project = QgsProject.instance()
        # Перебор всех слоев в проекте
        for layer in list(project.mapLayers().values()):
            # Проверка, содержит ли имя слоя слово "buffer"
            if "buffer" in layer.name().lower():
                # Удалить слой из проекта
                project.removeMapLayer(layer)

        self._delete_files()
        # Очистка кэша холста карты
        iface.mapCanvas().refreshAllLayers()
        # Очистка кэша рендеринга
        project.reloadAllLayers()

    def prepare(self) -> None:
        project = QgsProject.instance()
        # Удалить все слои
        for layer in list(project.mapLayers().values()):
            project.removeMapLayer(layer)

        # Удалить определенные форматы файлов (e.g., shapefiles, GeoTIFFs, etc.)
        self._delete_files()
        # Очистка кэша холста карты
        iface.mapCanvas().refreshAllLayers()
        # Очистка кэша рендеринга
        project.reloadAllLayers()

    def _delete_files(self) -> None:
        """Удалить определенные форматы файлов (e.g., shapefiles, GeoTIFFs, etc.)."""
        file_patterns = ["*.shp", "*.shx", "*.dbf", "*.prj", "*.tif"]
        files_to_delete: list[Path] = []
        for pattern in file_patterns:
            files_to_delete.extend(Path(self.project_folder).glob(pattern))

        for file_path in files_to_delete:
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error deleting {file_path}: {e}", flush=True)
            else:
                print(f"Deleted: {file_path}", flush=True)

    def run_programm(self) -> None:
        # Подготовка к работе
        self.clear_cache()
        self.prepare()

        # Запуск диалогового окна
        self.show_choice_dialog()
        self.show_layer_visibility_dialog()
