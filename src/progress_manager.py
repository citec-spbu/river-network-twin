# progress_manager.py
from qgis.PyQt.QtWidgets import QProgressDialog
from qgis.PyQt.QtCore import Qt, QTimer


class ProgressManager:
    def __init__(self, title="Выполнение операции", label="Обработка..."):
        self.progress = None
        self.title = title
        self.initial_label = label
        self._force_show = False
        self._keep_active_timer = QTimer()
        self._keep_active_timer.timeout.connect(self._keep_active)

    def init_progress(self, maximum=100):
        """Инициализирует диалог прогресса"""
        self.progress = QProgressDialog(self.initial_label, "Отмена", 0, maximum)
        self.progress.setWindowTitle(self.title)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self._force_show = True
        self._keep_active_timer.start(100)  # Таймер для периодической активации
        self._keep_active()

    def _keep_active(self):
        """Поддерживает диалог видимым и активным"""
        if self.progress and self._force_show:
            self.progress.show()
            self.progress.raise_()
            self.progress.activateWindow()
            from qgis.PyQt.QtWidgets import QApplication
            QApplication.processEvents()

    def update(self, value, label=None):
        """Обновляет прогресс"""
        if self.progress is None:
            self.init_progress()

        if label:
            self.progress.setLabelText(label)
        self.progress.setValue(value)
        self._keep_active()

        # Даем возможность обработать события GUI
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.processEvents()

        return not self.progress.wasCanceled()

    def was_canceled(self):
        """Проверяет, была ли операция отменена"""
        return self.progress.wasCanceled() if self.progress else False

    def finish(self):
        """Завершает работу с прогрессом"""
        if self.progress:
            self._keep_active_timer.stop()
            self.progress.close()
