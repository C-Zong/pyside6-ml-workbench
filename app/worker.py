from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .data_ops import load_table
from .ml_ops import train_model


class LoadWorker(QObject):
    """Load one dataset in a background thread."""

    finished = Signal(str, object)
    error = Signal(str, str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    @Slot()
    def run(self):
        file_name = Path(self.file_path).name
        try:
            self.finished.emit(file_name, load_table(self.file_path))
        except Exception as exc:
            self.error.emit(file_name, str(exc))


class TrainWorker(QObject):
    """Train one model in a background thread."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, train_df, eval_df, features, target, model_name, task_type, continue_bundle=None):
        super().__init__()
        self.train_df = train_df
        self.eval_df = eval_df
        self.features = features
        self.target = target
        self.model_name = model_name
        self.task_type = task_type
        self.continue_bundle = continue_bundle

    @Slot()
    def run(self):
        try:
            result = train_model(
                train_df=self.train_df,
                eval_df=self.eval_df,
                features=self.features,
                target=self.target,
                model_name=self.model_name,
                task_type=self.task_type,
                continue_bundle=self.continue_bundle,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
