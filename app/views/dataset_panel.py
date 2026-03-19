from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QListWidget, QMenu, QPushButton, QVBoxLayout, QWidget


class DatasetPanel(QWidget):
    """Shared left panel for loading files and choosing the active dataset."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.load_btn = QPushButton("Load Excel/CSV")
        self.load_model_btn = QPushButton("Load Model")
        self.dataset_list = QListWidget()
        self.dataset_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.trained_model_list = QListWidget()
        self.trained_model_list.setContextMenuPolicy(Qt.CustomContextMenu)

        layout.addWidget(self.load_btn)
        layout.addWidget(QLabel("Loaded Datasets"))
        layout.addWidget(self.dataset_list)
        layout.addWidget(self.load_model_btn)
        layout.addWidget(QLabel("Trained Model"))
        layout.addWidget(self.trained_model_list)

    def set_trained_model_items(self, names: list[str], current_name: str | None = None) -> None:
        """Refresh the selectable trained-model list."""
        self.trained_model_list.blockSignals(True)
        self.trained_model_list.clear()
        for name in names:
            self.trained_model_list.addItem(name)
        if current_name:
            for index in range(self.trained_model_list.count()):
                item = self.trained_model_list.item(index)
                if item is not None and item.text() == current_name:
                    self.trained_model_list.setCurrentItem(item)
                    break
        self.trained_model_list.blockSignals(False)

    def show_dataset_context_menu(self, position) -> tuple[str | None, str | None]:
        """Return the dataset action and name chosen from the context menu."""
        item = self.dataset_list.itemAt(position)
        if item is None:
            return None, None

        menu = QMenu(self)
        export_action = menu.addAction("Export")
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(self.dataset_list.mapToGlobal(position))
        if chosen == export_action:
            return "export", item.text()
        if chosen == delete_action:
            return "delete", item.text()
        return None, None

    def show_model_context_menu(self, position) -> tuple[str | None, str | None]:
        """Return the trained-model action and model name chosen from the context menu."""
        item = self.trained_model_list.itemAt(position)
        if item is None:
            return None, None

        menu = QMenu(self)
        export_action = menu.addAction("Export")
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(self.trained_model_list.mapToGlobal(position))
        if chosen == export_action:
            return "export", item.text()
        if chosen == delete_action:
            return "delete", item.text()
        return None, None
