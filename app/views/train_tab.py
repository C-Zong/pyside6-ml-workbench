from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget


class TrainTab(QWidget):
    """Controls for model training."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.target_combo = QComboBox()
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.model_combo = QComboBox()
        self.incremental_checkbox = QCheckBox("Continue Incremental Model")
        self.incremental_checkbox.setEnabled(False)
        self.train_btn = QPushButton("Train Model")

        layout.addWidget(QLabel("Target Column"))
        layout.addWidget(self.target_combo)
        layout.addWidget(QLabel("Feature Columns"))
        layout.addWidget(self.feature_list)
        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.incremental_checkbox)
        layout.addWidget(self.train_btn)
        layout.addStretch()
