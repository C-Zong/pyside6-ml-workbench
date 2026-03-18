from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QObject, QThread, Signal

from .data_ops import (
    build_column_profile,
    clean_dataframe,
    create_formatted_column,
    create_time_range_dataset,
    delete_rows_by_spec,
    drop_bad_rows,
    drop_column,
    infer_task_type,
    merge_tables,
    replace_column_with_zero,
    save_table,
    split_column_by_delimiter_occurrence,
)
from .models import get_model_options, is_incremental_model
from .plotting import build_distribution_figure, build_importance_figure
from .state import AppState
from .worker import LoadWorker, TrainWorker


class AppController(QObject):
    """Coordinate data workflows and emit UI-ready state updates."""

    datasets_changed = Signal(list)
    columns_changed = Signal(list)
    working_df_changed = Signal(object)
    clean_profile_changed = Signal(str)
    clean_figure_changed = Signal(object)
    model_options_changed = Signal(list)
    training_results_changed = Signal(str, object)
    load_button_enabled = Signal(bool)
    train_button_enabled = Signal(bool)
    status_changed = Signal(str)
    error_occurred = Signal(str, str)
    trained_model_changed = Signal(str)

    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        self.load_threads: list[QThread] = []
        self.load_workers: dict[QThread, LoadWorker] = {}
        self.train_thread: QThread | None = None
        self.train_worker: TrainWorker | None = None

    def load_files(self, files: list[str]) -> None:
        """Load datasets in parallel background threads."""
        if not files:
            return

        self.load_threads.clear()
        self.load_workers.clear()
        self.load_button_enabled.emit(False)
        self.status_changed.emit(f"Loading {len(files)} file(s)...")

        for file_path in files:
            thread = QThread()
            worker = LoadWorker(file_path)
            self.load_workers[thread] = worker
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._on_load_finished)
            worker.error.connect(self._on_load_error)
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(lambda t=thread: self._drop_load_thread(t))
            self.load_threads.append(thread)
            thread.start()

    def select_dataset(self, dataset_name: str) -> None:
        """Set the selected dataset as the current working dataframe."""
        if dataset_name not in self.state.dataframes:
            return

        self.state.active_dataset_name = dataset_name
        self.state.working_df = self.state.dataframes[dataset_name].copy()
        self._emit_working_df()
        self.status_changed.emit(f"Selected dataset: {dataset_name}")

    def clean_current_dataset(self) -> None:
        """Run the default dataframe cleaning pipeline on the working dataset."""
        df = self._require_working_df()
        if df is None:
            return

        self.state.working_df = clean_dataframe(df)
        self._emit_working_df()
        self.status_changed.emit("Applied default cleaning.")

    def update_clean_column(self, column: str) -> None:
        """Refresh the clean-tab profile for a selected column."""
        df = self._require_working_df()
        if df is None or not column:
            return

        profile = build_column_profile(df, column)
        profile_lines = [
            f"Column: {profile['column']}",
            f"Dtype: {profile['dtype']}",
            f"Rows: {profile['rows']}",
            f"Zero ratio: {profile['zero_ratio']:.2%}",
            f"Empty string ratio: {profile['empty_ratio']:.2%}",
            f"NA ratio: {profile['na_ratio']:.2%}",
            f"Unique values: {profile['unique_count']}",
            "Top values:",
        ]
        profile_lines.extend(f"{key}: {value}" for key, value in profile["top_values"].items())
        self.clean_profile_changed.emit("\n".join(profile_lines))
        self.clean_figure_changed.emit(build_distribution_figure(df, column))

    def drop_selected_column(self, column: str) -> None:
        """Delete the chosen column from the working dataframe."""
        df = self._require_working_df()
        if df is None or not column:
            return

        self.state.working_df = drop_column(df, column)
        self._emit_working_df()
        self.clean_profile_changed.emit("")
        self.clean_figure_changed.emit(None)
        self.status_changed.emit(f"Deleted column: {column}")

    def zero_selected_column(self, column: str) -> None:
        """Replace all values in the chosen column with zero."""
        df = self._require_working_df()
        if df is None or not column:
            return

        self.state.working_df = replace_column_with_zero(df, column)
        self._emit_working_df()
        self.update_clean_column(column)
        self.status_changed.emit(f"Set column to zero: {column}")

    def export_current_dataset(self, file_path: str) -> None:
        """Export the current working dataframe."""
        df = self._require_working_df()
        if df is None:
            return

        try:
            save_table(df, file_path)
        except Exception as exc:
            self.error_occurred.emit("Export error", str(exc))
            return

        self.status_changed.emit(f"Exported dataset to {file_path}")

    def delete_rows(self, spec: str) -> None:
        """Delete rows by explicit index specification."""
        df = self._require_working_df()
        if df is None:
            return

        try:
            self.state.working_df = delete_rows_by_spec(df, spec)
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._emit_working_df()
        self.status_changed.emit("Deleted selected rows.")

    def delete_bad_rows(self, columns: list[str], threshold: float) -> None:
        """Delete rows with a high ratio of NA/0/empty values."""
        df = self._require_working_df()
        if df is None:
            return

        try:
            target_columns = columns or list(df.columns)
            self.state.working_df = drop_bad_rows(df, target_columns, threshold)
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._emit_working_df()
        self.status_changed.emit("Removed rows with high bad-value ratios.")

    def create_range_dataset(
        self,
        time_column: str,
        start_text: str,
        end_text: str,
        dataset_name: str,
    ) -> None:
        """Create a new loaded dataset from the current working dataset time range."""
        df = self._require_working_df()
        if df is None:
            return

        try:
            ranged_df = create_time_range_dataset(df, time_column, start_text, end_text)
        except Exception as exc:
            self.error_occurred.emit("Range dataset error", str(exc))
            return

        result_name = self._build_range_dataset_name(dataset_name, time_column, start_text, end_text)
        self.state.dataframes[result_name] = ranged_df
        self.state.active_dataset_name = result_name
        self.state.working_df = ranged_df.copy()
        self._emit_datasets()
        self._emit_working_df()
        self.status_changed.emit(f"Created range dataset: {result_name}")

    def create_column(
        self,
        new_column: str,
        left_column: str,
        right_column: str,
        separator: str,
        left_width: int,
        right_width: int,
    ) -> None:
        """Create a new formatted text column in the working dataframe."""
        df = self._require_working_df()
        if df is None:
            return

        try:
            self.state.working_df = create_formatted_column(
                df,
                new_column=new_column,
                left_column=left_column,
                right_column=right_column,
                separator=separator,
                left_width=left_width,
                right_width=right_width,
            )
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._emit_working_df()
        self.status_changed.emit(f"Created column: {new_column}")

    def split_column_by_delimiter(
        self,
        source_column: str,
        left_new_column: str,
        right_new_column: str,
        delimiter: str,
        occurrence: int,
    ) -> None:
        """Split one source column into two columns at a chosen delimiter occurrence."""
        df = self._require_working_df()
        if df is None or not source_column:
            return

        try:
            self.state.working_df = split_column_by_delimiter_occurrence(
                df,
                source_column=source_column,
                left_new_column=left_new_column,
                right_new_column=right_new_column,
                delimiter=delimiter,
                occurrence=occurrence,
            )
        except Exception as exc:
            self.error_occurred.emit("Column split error", str(exc))
            return

        self._emit_working_df()
        self.status_changed.emit(f"Split column by delimiter: {source_column}")

    def merge_datasets(
        self,
        left_name: str,
        right_name: str,
        left_key: str,
        right_key: str,
        join_type: str,
    ) -> None:
        """Merge or append two loaded datasets and register the result as a new dataset."""
        if left_name not in self.state.dataframes or right_name not in self.state.dataframes:
            self.error_occurred.emit("Merge error", "Please choose two valid datasets.")
            return

        try:
            merged = merge_tables(
                self.state.dataframes[left_name],
                self.state.dataframes[right_name],
                left_key=left_key,
                right_key=right_key,
                how=join_type,
            )
        except Exception as exc:
            self.error_occurred.emit("Merge error", str(exc))
            return

        result_name = self._build_result_name(left_name, right_name, join_type)
        self.state.dataframes[result_name] = merged
        self.state.active_dataset_name = result_name
        self.state.working_df = merged.copy()
        self._emit_datasets()
        self._emit_working_df()
        self.status_changed.emit(f"Created merged dataset: {result_name}")

    def update_target(self, target: str) -> None:
        """Infer task type for the selected target and refresh model options."""
        df = self._require_working_df()
        if df is None or not target:
            return

        self.state.task_type = infer_task_type(df, target)
        self.model_options_changed.emit(get_model_options(self.state.task_type))

    def train_model(
        self,
        target: str,
        features: list[str],
        model_name: str,
        continue_training: bool = False,
    ) -> None:
        """Start model training on the current working dataset."""
        df = self._require_working_df()
        if df is None:
            return
        if not target or not features:
            self.error_occurred.emit("Train error", "Please choose a target and at least one feature.")
            return
        if continue_training and not is_incremental_model(model_name):
            self.error_occurred.emit("Train error", "Continue Incremental Model only works with incremental models.")
            return
        if continue_training and not self.can_continue_training(target, features, model_name):
            self.error_occurred.emit("Train error", "No compatible incremental model state is available to continue.")
            return

        self.state.target_column = target
        self.state.feature_columns = features
        self.state.selected_model_name = model_name

        self.train_button_enabled.emit(False)
        self.status_changed.emit("Training started...")

        self.train_thread = QThread()
        self.train_worker = TrainWorker(
            train_df=df.copy(),
            eval_df=None,
            features=features,
            target=target,
            model_name=model_name,
            task_type=self.state.task_type,
            continue_bundle=self.state.incremental_model_bundle if continue_training else None,
        )
        self.train_worker.moveToThread(self.train_thread)
        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.finished.connect(self.train_thread.quit)
        self.train_worker.error.connect(self.train_thread.quit)
        self.train_thread.finished.connect(self.train_thread.deleteLater)
        self.train_thread.start()

    def _require_working_df(self):
        if self.state.working_df is None:
            self.error_occurred.emit("Dataset error", "Please load and select a dataset first.")
        return self.state.working_df

    def _emit_working_df(self) -> None:
        if self.state.working_df is None:
            return
        columns = [str(column) for column in self.state.working_df.columns]
        self.columns_changed.emit(columns)
        self.working_df_changed.emit(self.state.working_df.copy())

    def _emit_datasets(self) -> None:
        self.datasets_changed.emit(list(self.state.dataframes.keys()))

    def _on_load_finished(self, file_name: str, df) -> None:
        self.state.dataframes[file_name] = df
        if self.state.active_dataset_name is None:
            self.state.active_dataset_name = file_name
            self.state.working_df = df.copy()
            self._emit_working_df()
        self._emit_datasets()

    def _on_load_error(self, file_name: str, message: str) -> None:
        self.error_occurred.emit("Load error", f"Failed to load {file_name}\n{message}")

    def _drop_load_thread(self, thread: QThread) -> None:
        self.load_workers.pop(thread, None)
        self.load_threads = [active_thread for active_thread in self.load_threads if active_thread is not thread]
        if not self.load_threads:
            self.load_button_enabled.emit(True)
            self.status_changed.emit(f"Loaded {len(self.state.dataframes)} dataset(s).")

    def _on_train_finished(self, result) -> None:
        self.train_button_enabled.emit(True)
        self.train_worker = None
        self.train_thread = None

        self.state.trained_model = result.pipeline if result.pipeline is not None else result.model_bundle
        self.state.metrics = result.metrics
        self.state.feature_importance = result.importance_df
        if result.model_bundle is not None:
            self.state.incremental_model_bundle = result.model_bundle
            self.state.incremental_features = list(self.state.feature_columns)
            self.state.incremental_target = self.state.target_column
            self.state.incremental_model_name = self.state.selected_model_name
            self.state.incremental_task_type = self.state.task_type
        elif not is_incremental_model(self.state.selected_model_name):
            self.state.incremental_model_bundle = None
            self.state.incremental_features = []
            self.state.incremental_target = None
            self.state.incremental_model_name = None
            self.state.incremental_task_type = None

        metric_lines = [f"{key}: {value:.4f}" for key, value in result.metrics.items()]
        metrics_text = "Training complete\n" + "\n".join(metric_lines)
        figure = build_importance_figure(result.importance_df) if not result.importance_df.empty else None
        self.training_results_changed.emit(metrics_text, figure)
        self.trained_model_changed.emit(self.describe_trained_model())
        self.status_changed.emit("Training complete.")

    def _on_train_error(self, message: str) -> None:
        self.train_button_enabled.emit(True)
        self.train_worker = None
        self.train_thread = None
        self.error_occurred.emit("Training error", message)

    def _build_result_name(self, left_name: str, right_name: str, join_type: str) -> str:
        base_name = f"Merged {join_type.title()} {Path(left_name).stem} + {Path(right_name).stem}"
        candidate = base_name
        suffix = 2
        while candidate in self.state.dataframes:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def can_continue_training(self, target: str, features: list[str], model_name: str) -> bool:
        """Return whether the current selections match a resumable incremental model."""
        return (
            self.state.incremental_model_bundle is not None
            and self.state.incremental_target == target
            and self.state.incremental_features == list(features)
            and self.state.incremental_model_name == model_name
            and self.state.incremental_task_type == self.state.task_type
        )

    def describe_trained_model(self) -> str:
        """Build a short summary for the left-side trained-model panel."""
        if self.state.trained_model is None:
            return "No Trained Model"
        summary = [
            self.state.selected_model_name or "Model",
            f"Target: {self.state.target_column or '-'}",
            f"Features: {len(self.state.feature_columns)}",
        ]
        if self.state.incremental_model_bundle is not None:
            summary.append("Incremental Ready")
        return "\n".join(summary)

    def _build_range_dataset_name(self, dataset_name: str, time_column: str, start_text: str, end_text: str) -> str:
        base_name = dataset_name.strip() or self._default_range_dataset_name(time_column, start_text, end_text)
        candidate = base_name
        suffix = 2
        while candidate in self.state.dataframes:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _default_range_dataset_name(self, time_column: str, start_text: str, end_text: str) -> str:
        active_name = self.state.active_dataset_name or "Dataset"
        start_part = start_text.strip() or "Start"
        end_part = end_text.strip() or "End"
        return f"{Path(active_name).stem} {time_column} {start_part} To {end_part}"
