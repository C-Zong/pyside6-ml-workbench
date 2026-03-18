from PySide6.QtWidgets import QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget


class DatasetPanel(QWidget):
    """Shared left panel for loading files and choosing the active dataset."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.load_btn = QPushButton("Load Excel/CSV")
        self.dataset_list = QListWidget()
        self.trained_model_label = QLabel("No Trained Model")
        self.trained_model_label.setWordWrap(True)

        layout.addWidget(self.load_btn)
        layout.addWidget(QLabel("Loaded Datasets"))
        layout.addWidget(self.dataset_list)
        layout.addWidget(QLabel("Trained Model"))
        layout.addWidget(self.trained_model_label)
