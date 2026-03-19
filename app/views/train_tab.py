from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainTab(QWidget):
    """Controls for model training."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.target_combo = QComboBox()
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItem("Auto", "")
        self.task_type_combo.addItem("Classification", "classification")
        self.task_type_combo.addItem("Regression", "regression")
        self.model_combo = QComboBox()
        self.params_box = QWidget()
        self.params_form = QFormLayout(self.params_box)
        self.rf_estimators_spin = QSpinBox()
        self.rf_estimators_spin.setRange(10, 2000)
        self.rf_estimators_spin.setValue(100)
        self.rf_max_depth_spin = QSpinBox()
        self.rf_max_depth_spin.setRange(0, 200)
        self.rf_max_depth_spin.setSpecialValueText("None")
        self.rf_max_depth_spin.setValue(0)
        self.gb_estimators_spin = QSpinBox()
        self.gb_estimators_spin.setRange(10, 2000)
        self.gb_estimators_spin.setValue(100)
        self.gb_learning_rate_spin = QDoubleSpinBox()
        self.gb_learning_rate_spin.setRange(0.001, 10.0)
        self.gb_learning_rate_spin.setDecimals(3)
        self.gb_learning_rate_spin.setSingleStep(0.01)
        self.gb_learning_rate_spin.setValue(0.1)
        self.linear_c_spin = QDoubleSpinBox()
        self.linear_c_spin.setRange(0.001, 1000.0)
        self.linear_c_spin.setDecimals(3)
        self.linear_c_spin.setSingleStep(0.1)
        self.linear_c_spin.setValue(1.0)
        self.sgd_alpha_spin = QDoubleSpinBox()
        self.sgd_alpha_spin.setRange(0.000001, 1.0)
        self.sgd_alpha_spin.setDecimals(6)
        self.sgd_alpha_spin.setSingleStep(0.0001)
        self.sgd_alpha_spin.setValue(0.0001)
        self.sgd_max_iter_spin = QSpinBox()
        self.sgd_max_iter_spin.setRange(100, 100000)
        self.sgd_max_iter_spin.setValue(1000)
        self.incremental_checkbox = QCheckBox("Continue Incremental Model")
        self.incremental_checkbox.setEnabled(False)
        self.train_btn = QPushButton("Train Model")
        self.predict_btn = QPushButton("Predict Current Dataset")
        self.predict_btn.setEnabled(False)

        self._param_rows = [
            ("rf_estimators", "Number Of Trees", self.rf_estimators_spin),
            ("rf_max_depth", "Max Depth", self.rf_max_depth_spin),
            ("gb_estimators", "Number Of Estimators", self.gb_estimators_spin),
            ("gb_learning_rate", "Learning Rate", self.gb_learning_rate_spin),
            ("linear_c", "C", self.linear_c_spin),
            ("sgd_alpha", "Alpha", self.sgd_alpha_spin),
            ("sgd_max_iter", "Max Iterations", self.sgd_max_iter_spin),
        ]
        for _, label, widget in self._param_rows:
            self.params_form.addRow(label, widget)

        layout.addWidget(QLabel("Target Column"))
        layout.addWidget(self.target_combo)
        layout.addWidget(QLabel("Task Type"))
        layout.addWidget(self.task_type_combo)
        layout.addWidget(QLabel("Feature Columns"))
        layout.addWidget(self.feature_list)
        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Model Parameters"))
        layout.addWidget(self.params_box)
        layout.addWidget(self.incremental_checkbox)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.predict_btn)
        layout.addStretch()

        self.update_model_params("RandomForestClassifier")

    def update_model_params(self, model_name: str) -> None:
        """Show only the key parameter controls for the selected model."""
        visible_keys = {
            "RandomForestClassifier": {"rf_estimators", "rf_max_depth"},
            "RandomForestRegressor": {"rf_estimators", "rf_max_depth"},
            "GradientBoostingClassifier": {"gb_estimators", "gb_learning_rate"},
            "GradientBoostingRegressor": {"gb_estimators", "gb_learning_rate"},
            "LogisticRegression": {"linear_c"},
            "SVC": {"linear_c"},
            "SVR": {"linear_c"},
            "SGDClassifier": {"sgd_alpha", "sgd_max_iter"},
            "SGDRegressor": {"sgd_alpha", "sgd_max_iter"},
        }.get(model_name, set())

        for key, _, widget in self._param_rows:
            label = self.params_form.labelForField(widget)
            is_visible = key in visible_keys
            if label is not None:
                label.setVisible(is_visible)
            widget.setVisible(is_visible)
        self.params_box.setVisible(bool(visible_keys))

    def get_model_params(self, model_name: str) -> dict:
        """Collect the active model parameters from visible controls."""
        if model_name in {"RandomForestClassifier", "RandomForestRegressor"}:
            return {
                "n_estimators": self.rf_estimators_spin.value(),
                "max_depth": self.rf_max_depth_spin.value() or None,
            }
        if model_name in {"GradientBoostingClassifier", "GradientBoostingRegressor"}:
            return {
                "n_estimators": self.gb_estimators_spin.value(),
                "learning_rate": self.gb_learning_rate_spin.value(),
            }
        if model_name in {"LogisticRegression", "SVC", "SVR"}:
            return {"C": self.linear_c_spin.value()}
        if model_name in {"SGDClassifier", "SGDRegressor"}:
            return {
                "alpha": self.sgd_alpha_spin.value(),
                "max_iter": self.sgd_max_iter_spin.value(),
            }
        return {}

    def selected_task_type(self) -> str | None:
        """Return the chosen task type, or None when Auto is active."""
        return self.task_type_combo.currentData() or None
