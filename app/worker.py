from PySide6.QtCore import QObject, Signal, Slot
from .ml_ops import train_model


class TrainWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, df, features, target, model_name, task_type):
        super().__init__()
        self.df = df
        self.features = features
        self.target = target
        self.model_name = model_name
        self.task_type = task_type

    @Slot()
    def run(self):
        try:
            result = train_model(
                df=self.df,
                features=self.features,
                target=self.target,
                model_name=self.model_name,
                task_type=self.task_type,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))