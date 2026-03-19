from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .data_ops import aggregate_dataset, coalesce_merged_columns, convert_aggregate_column_to_binary, load_table, merge_tables
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

    def __init__(self, train_df, eval_df, features, target, model_name, task_type, continue_bundle=None, model_params=None):
        super().__init__()
        self.train_df = train_df
        self.eval_df = eval_df
        self.features = features
        self.target = target
        self.model_name = model_name
        self.task_type = task_type
        self.continue_bundle = continue_bundle
        self.model_params = model_params or {}

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
                model_params=self.model_params,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MergeWorker(QObject):
    """Merge datasets in a background thread."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        left_df,
        right_df,
        left_key,
        right_key,
        join_type,
        aggregate_right=False,
        aggregate_specs=None,
        coalesce_columns=False,
        coalesce_strategy="first_non_empty",
    ):
        super().__init__()
        self.left_df = left_df
        self.right_df = right_df
        self.left_key = left_key
        self.right_key = right_key
        self.join_type = join_type
        self.aggregate_right = aggregate_right
        self.aggregate_specs = aggregate_specs or []
        self.coalesce_columns = coalesce_columns
        self.coalesce_strategy = coalesce_strategy

    @Slot()
    def run(self):
        try:
            right_df = self.right_df
            if self.aggregate_right:
                right_df = aggregate_dataset(
                    self.right_df,
                    group_key=self.right_key,
                    specs=self.aggregate_specs,
                )
            merged = merge_tables(
                self.left_df,
                right_df,
                left_key=self.left_key,
                right_key=self.right_key,
                how=self.join_type,
            )
            if self.coalesce_columns and self.join_type != "append":
                merged = coalesce_merged_columns(
                    merged,
                    strategy=self.coalesce_strategy,
                    left_key=self.left_key,
                    right_key=self.right_key,
                )
            self.finished.emit(merged)
        except Exception as exc:
            self.error.emit(str(exc))


class AggregateWorker(QObject):
    """Aggregate one dataset in a background thread."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, df, group_key, specs, binary_config=None):
        super().__init__()
        self.df = df
        self.group_key = group_key
        self.specs = specs
        self.binary_config = binary_config or None

    @Slot()
    def run(self):
        try:
            aggregated = aggregate_dataset(self.df, self.group_key, self.specs)
            if self.binary_config:
                aggregated = convert_aggregate_column_to_binary(
                    aggregated,
                    source_column=self.binary_config["source_column"],
                    threshold=self.binary_config["threshold"],
                    output_column=self.binary_config.get("output_column", ""),
                )
            self.finished.emit(aggregated)
        except Exception as exc:
            self.error.emit(str(exc))
