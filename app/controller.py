from __future__ import annotations

import pickle
from pathlib import Path
import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal

from .data_ops import (
    build_column_profile,
    create_formatted_column,
    create_time_range_dataset,
    delete_rows_by_spec,
    drop_bad_rows,
    drop_columns,
    infer_task_type,
    replace_bad_values_with_zero,
    save_table,
    summarize_merge_risk,
    split_column_by_delimiter_occurrence,
)
from .models import get_model_options, is_incremental_model
from .ml_ops import predict_dataframe
from .plotting import build_distribution_figure, build_importance_figure
from .state import AppState
from .worker import AggregateWorker, LoadWorker, MergeWorker, TrainWorker


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
    aggregate_button_enabled = Signal(bool)
    merge_button_enabled = Signal(bool)
    train_button_enabled = Signal(bool)
    status_changed = Signal(str)
    error_occurred = Signal(str, str)
    trained_model_changed = Signal(str)
    trained_models_changed = Signal(list, str)
    merge_risk_changed = Signal(str)

    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        self.load_threads: list[QThread] = []
        self.load_workers: dict[QThread, LoadWorker] = {}
        self.train_thread: QThread | None = None
        self.train_worker: TrainWorker | None = None
        self.aggregate_thread: QThread | None = None
        self.aggregate_worker: AggregateWorker | None = None
        self.merge_thread: QThread | None = None
        self.merge_worker: MergeWorker | None = None

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

        self._activate_dataset(dataset_name, self.state.dataframes[dataset_name])
        self.status_changed.emit(f"Selected dataset: {dataset_name}")

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete one loaded dataset and select a replacement when needed."""
        if dataset_name not in self.state.dataframes:
            return

        del self.state.dataframes[dataset_name]
        if self.state.active_dataset_name == dataset_name:
            remaining_names = list(self.state.dataframes.keys())
            if remaining_names:
                next_name = remaining_names[0]
                self._activate_dataset(next_name, self.state.dataframes[next_name])
            else:
                self.state.active_dataset_name = None
                self._set_working_df(pd.DataFrame())
        self._emit_datasets()
        self.status_changed.emit(f"Deleted dataset: {dataset_name}")

    def export_dataset(self, dataset_name: str, file_path: str) -> None:
        """Export one loaded dataset by name."""
        df = self.state.dataframes.get(dataset_name)
        if df is None:
            self.error_occurred.emit("Export error", "Selected dataset no longer exists.")
            return

        try:
            save_table(df, file_path)
        except Exception as exc:
            self.error_occurred.emit("Export error", str(exc))
            return

        self.status_changed.emit(f"Exported dataset to {file_path}")

    def export_trained_model(self, file_path: str, model_name: str | None = None) -> None:
        """Export the stored trained model bundle or pipeline."""
        entry = self._get_model_entry(model_name)
        if entry is None:
            self.error_occurred.emit("Export error", "No trained model is available.")
            return

        try:
            with open(file_path, "wb") as handle:
                pickle.dump(
                    {
                        "trained_model": entry["trained_model"],
                        "feature_columns": list(entry.get("feature_columns") or []),
                        "target_column": entry.get("target_column"),
                        "selected_model_name": entry.get("selected_model_name"),
                        "task_type": entry.get("task_type"),
                        "metrics": dict(entry.get("metrics") or {}),
                        "feature_importance": entry.get("feature_importance"),
                    },
                    handle,
                )
        except Exception as exc:
            self.error_occurred.emit("Export error", str(exc))
            return

        self.status_changed.emit(f"Exported trained model to {file_path}")

    def load_trained_model(self, file_path: str) -> None:
        """Load one previously exported model file."""
        try:
            with open(file_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            self.error_occurred.emit("Load model error", str(exc))
            return

        if isinstance(payload, dict) and "trained_model" in payload:
            entry = {
                "trained_model": payload.get("trained_model"),
                "feature_columns": list(payload.get("feature_columns") or []),
                "target_column": payload.get("target_column"),
                "selected_model_name": payload.get("selected_model_name") or "Loaded Model",
                "task_type": payload.get("task_type") or self.state.task_type,
                "metrics": dict(payload.get("metrics") or {}),
                "feature_importance": payload.get("feature_importance"),
                "incremental_model_bundle": None,
                "incremental_features": [],
                "incremental_target": None,
                "incremental_model_name": None,
                "incremental_task_type": None,
            }
        else:
            entry = {
                "trained_model": payload,
                "feature_columns": [],
                "target_column": None,
                "selected_model_name": "Loaded Model",
                "task_type": self.state.task_type,
                "metrics": {},
                "feature_importance": None,
                "incremental_model_bundle": None,
                "incremental_features": [],
                "incremental_target": None,
                "incremental_model_name": None,
                "incremental_task_type": None,
            }

        model_name = self._build_model_entry_name(Path(file_path).stem)
        self.state.trained_models[model_name] = entry
        self.select_trained_model(model_name)
        self.status_changed.emit(f"Loaded model: {Path(file_path).name}")

    def select_trained_model(self, model_name: str) -> None:
        """Activate one model entry from the left-side model list."""
        if model_name == self.state.active_model_name:
            return

        entry = self.state.trained_models.get(model_name)
        if entry is None:
            return

        self.state.active_model_name = model_name
        self.state.trained_model = entry.get("trained_model")
        self.state.feature_columns = list(entry.get("feature_columns") or [])
        self.state.target_column = entry.get("target_column")
        self.state.selected_model_name = entry.get("selected_model_name") or self.state.selected_model_name
        self.state.task_type = entry.get("task_type") or self.state.task_type
        self.state.metrics = dict(entry.get("metrics") or {})
        self.state.feature_importance = entry.get("feature_importance")
        self.state.incremental_model_bundle = entry.get("incremental_model_bundle")
        self.state.incremental_features = list(entry.get("incremental_features") or [])
        self.state.incremental_target = entry.get("incremental_target")
        self.state.incremental_model_name = entry.get("incremental_model_name")
        self.state.incremental_task_type = entry.get("incremental_task_type")

        figure = build_importance_figure(self.state.feature_importance) if self.state.feature_importance is not None and not self.state.feature_importance.empty else None
        metrics_text = ""
        if self.state.metrics:
            metrics_text = "Training complete\n" + "\n".join(f"{key}: {value:.4f}" for key, value in self.state.metrics.items())
        self.training_results_changed.emit(metrics_text, figure)
        self._emit_trained_models()

    def delete_trained_model(self, model_name: str) -> None:
        """Delete one saved model entry."""
        if model_name not in self.state.trained_models:
            return

        del self.state.trained_models[model_name]
        if self.state.active_model_name == model_name:
            remaining = list(self.state.trained_models.keys())
            if remaining:
                self.select_trained_model(remaining[0])
            else:
                self.state.active_model_name = None
                self._clear_active_model_state()
                self.training_results_changed.emit("", None)
                self._emit_trained_models()
        else:
            self._emit_trained_models()
        self.status_changed.emit(f"Deleted model: {model_name}")

    def predict_current_dataset(self) -> None:
        """Generate predictions for the active dataset and save them as a new dataset."""
        df = self._require_working_df()
        if df is None:
            return
        if self.state.trained_model is None:
            self.error_occurred.emit("Predict error", "Please train or load a model first.")
            return
        if not self.state.feature_columns:
            self.error_occurred.emit("Predict error", "This model does not have saved feature metadata.")
            return

        missing = [feature for feature in self.state.feature_columns if feature not in df.columns]
        if missing:
            self.error_occurred.emit("Predict error", f"Missing feature columns: {', '.join(missing[:5])}")
            return

        try:
            predictions = predict_dataframe(self.state.trained_model, df, self.state.feature_columns)
        except Exception as exc:
            self.error_occurred.emit("Predict error", str(exc))
            return

        predicted_df = df.copy()
        prediction_column = f"{self.state.target_column or 'Prediction'} Prediction"
        predicted_df[prediction_column] = predictions
        result_name = self._build_prediction_result_name()
        self.state.dataframes[result_name] = predicted_df
        self._activate_dataset(result_name, predicted_df, emit_datasets=True)
        self.status_changed.emit(f"Created prediction dataset: {result_name}")

    def update_clean_column(self, column: str) -> None:
        """Refresh the clean-tab profile for a selected column."""
        df = self._require_working_df()
        if df is None or not column:
            return

        self.state.clean_column = column
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

    def drop_selected_columns(self, columns: list[str]) -> None:
        """Delete the chosen columns from the working dataframe."""
        df = self._require_working_df()
        if df is None or not columns:
            return

        try:
            updated_df = drop_columns(df, columns)
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._set_working_df(updated_df)
        current_column = self.state.clean_column
        if current_column in columns:
            self.state.clean_column = None
            self.clean_profile_changed.emit("")
            self.clean_figure_changed.emit(None)
        elif current_column:
            self.update_clean_column(current_column)
        self.status_changed.emit(f"Deleted columns: {', '.join(columns)}")

    def zero_fill_bad_values(self, columns: list[str]) -> None:
        """Replace null-like values with zero in the selected columns."""
        df = self._require_working_df()
        if df is None or not columns:
            return

        try:
            updated_df = replace_bad_values_with_zero(df, columns)
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._set_working_df(updated_df, status_message=f"Zero-filled bad values in: {', '.join(columns)}")
        if self.state.clean_column in columns:
            self.update_clean_column(self.state.clean_column)

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
            updated_df = delete_rows_by_spec(df, spec)
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._set_working_df(updated_df, status_message="Deleted selected rows.")

    def delete_bad_rows(self, columns: list[str], threshold: float) -> None:
        """Delete rows with a high ratio of NA/0/empty values."""
        df = self._require_working_df()
        if df is None:
            return

        try:
            target_columns = columns or list(df.columns)
            updated_df = drop_bad_rows(df, target_columns, threshold)
        except Exception as exc:
            self.error_occurred.emit("Process error", str(exc))
            return

        self._set_working_df(updated_df, status_message="Removed rows with high bad-value ratios.")

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
        self._activate_dataset(result_name, ranged_df, emit_datasets=True)
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
            updated_df = create_formatted_column(
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

        self._set_working_df(updated_df, status_message=f"Created column: {new_column}")

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
            updated_df = split_column_by_delimiter_occurrence(
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

        self._set_working_df(updated_df, status_message=f"Split column by delimiter: {source_column}")

    def merge_datasets(
        self,
        left_name: str,
        right_name: str,
        left_key: str,
        right_key: str,
        join_type: str,
        aggregate_right: bool = False,
        aggregate_specs: list[dict] | None = None,
        coalesce_columns: bool = False,
        coalesce_strategy: str = "first_non_empty",
    ) -> None:
        """Merge or append two loaded datasets and register the result as a new dataset."""
        if left_name not in self.state.dataframes or right_name not in self.state.dataframes:
            self.error_occurred.emit("Merge error", "Please choose two valid datasets.")
            return
        aggregate_specs = aggregate_specs or []
        if aggregate_right and not aggregate_specs:
            self.error_occurred.emit("Merge error", "Please add at least one Aggregate Rule.")
            return

        self.merge_button_enabled.emit(False)
        self.status_changed.emit("Merge started...")
        result_name = self._build_result_name(left_name, right_name, join_type)

        self.merge_thread = QThread()
        self.merge_worker = MergeWorker(
            left_df=self.state.dataframes[left_name].copy(),
            right_df=self.state.dataframes[right_name].copy(),
            left_key=left_key,
            right_key=right_key,
            join_type=join_type,
            aggregate_right=aggregate_right,
            aggregate_specs=aggregate_specs,
            coalesce_columns=coalesce_columns,
            coalesce_strategy=coalesce_strategy,
        )
        self.merge_worker.moveToThread(self.merge_thread)
        self.merge_thread.started.connect(self.merge_worker.run)
        self.merge_worker.finished.connect(lambda merged_df, name=result_name: self._on_merge_finished(name, merged_df))
        self.merge_worker.error.connect(self._on_merge_error)
        self.merge_worker.finished.connect(self.merge_thread.quit)
        self.merge_worker.error.connect(self.merge_thread.quit)
        self.merge_thread.finished.connect(self.merge_thread.deleteLater)
        self.merge_thread.finished.connect(self._clear_merge_thread)
        self.merge_thread.start()

    def create_aggregated_dataset(
        self,
        dataset_name: str,
        group_key: str,
        specs: list[dict],
        output_name: str,
        binary_config: dict | None = None,
    ) -> None:
        """Aggregate one dataset into a new grouped dataset."""
        if dataset_name not in self.state.dataframes:
            self.error_occurred.emit("Aggregate error", "Please choose a valid dataset.")
            return
        if not specs:
            self.error_occurred.emit("Aggregate error", "Please add at least one Aggregate Rule.")
            return
        if binary_config:
            aggregate_columns = {spec.get("output_column", "").strip() for spec in specs}
            if binary_config["source_column"] not in aggregate_columns:
                self.error_occurred.emit("Aggregate error", "Binary Source Column must match an Aggregate Rule output.")
                return

        result_name = self._build_aggregate_result_name(dataset_name, output_name)
        self.aggregate_button_enabled.emit(False)
        self.status_changed.emit("Aggregation started...")

        self.aggregate_thread = QThread()
        self.aggregate_worker = AggregateWorker(
            df=self.state.dataframes[dataset_name].copy(),
            group_key=group_key,
            specs=specs,
            binary_config=binary_config,
        )
        self.aggregate_worker.moveToThread(self.aggregate_thread)
        self.aggregate_thread.started.connect(self.aggregate_worker.run)
        self.aggregate_worker.finished.connect(lambda aggregated_df, name=result_name: self._on_aggregate_finished(name, aggregated_df))
        self.aggregate_worker.error.connect(self._on_aggregate_error)
        self.aggregate_worker.finished.connect(self.aggregate_thread.quit)
        self.aggregate_worker.error.connect(self.aggregate_thread.quit)
        self.aggregate_thread.finished.connect(self.aggregate_thread.deleteLater)
        self.aggregate_thread.finished.connect(self._clear_aggregate_thread)
        self.aggregate_thread.start()

    def update_merge_risk(self, left_name: str, right_name: str, left_key: str, right_key: str) -> None:
        """Compute a quick merge-risk summary for the selected keys."""
        if left_name not in self.state.dataframes or right_name not in self.state.dataframes:
            self.merge_risk_changed.emit("")
            return
        if not left_key or not right_key:
            self.merge_risk_changed.emit("")
            return

        try:
            risk = summarize_merge_risk(
                self.state.dataframes[left_name],
                self.state.dataframes[right_name],
                left_key,
                right_key,
            )
        except Exception:
            self.merge_risk_changed.emit("")
            return

        lines = [
            f"Risk Level: {risk['risk_level']}",
            f"Left Rows: {risk['left_rows']}",
            f"Right Rows: {risk['right_rows']}",
            f"Left Unique Keys: {risk['left_unique']}",
            f"Right Unique Keys: {risk['right_unique']}",
            f"Left Duplicate Keys: {risk['left_duplicates']}",
            f"Right Duplicate Keys: {risk['right_duplicates']}",
        ]
        if risk["risk_level"] == "High":
            lines.append("Warning: both sides have duplicate keys, so merge size may explode.")
        self.merge_risk_changed.emit("\n".join(lines))

    def update_target(self, target: str, task_type_override: str | None = None) -> None:
        """Resolve task type for the selected target and refresh model options."""
        df = self._require_working_df()
        if df is None or not target:
            return

        self.state.task_type = task_type_override or infer_task_type(df, target)
        self.model_options_changed.emit(get_model_options(self.state.task_type))

    def train_model(
        self,
        target: str,
        features: list[str],
        model_name: str,
        model_params: dict | None = None,
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
            model_params=model_params or {},
        )
        self.train_worker.moveToThread(self.train_thread)
        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.finished.connect(self.train_thread.quit)
        self.train_worker.error.connect(self.train_thread.quit)
        self.train_thread.finished.connect(self.train_thread.deleteLater)
        self.train_thread.finished.connect(self._clear_train_thread)
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
            self._activate_dataset(file_name, df)
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

        metrics_text = self._build_metrics_text(result.metrics)
        figure = build_importance_figure(result.importance_df) if not result.importance_df.empty else None
        model_name = self._build_model_entry_name(self.state.selected_model_name or "Model")
        self.state.trained_models[model_name] = {
            "trained_model": self.state.trained_model,
            "feature_columns": list(self.state.feature_columns),
            "target_column": self.state.target_column,
            "selected_model_name": self.state.selected_model_name,
            "task_type": self.state.task_type,
            "metrics": dict(result.metrics),
            "feature_importance": result.importance_df.copy(),
            "incremental_model_bundle": self.state.incremental_model_bundle,
            "incremental_features": list(self.state.incremental_features),
            "incremental_target": self.state.incremental_target,
            "incremental_model_name": self.state.incremental_model_name,
            "incremental_task_type": self.state.incremental_task_type,
        }
        self.state.active_model_name = model_name
        self.training_results_changed.emit(metrics_text, figure)
        self._emit_trained_models()
        self.status_changed.emit("Training complete.")

    def _on_train_error(self, message: str) -> None:
        self.train_button_enabled.emit(True)
        self.error_occurred.emit("Training error", message)

    def _on_aggregate_finished(self, result_name: str, aggregated_df) -> None:
        self.aggregate_button_enabled.emit(True)
        self.state.dataframes[result_name] = aggregated_df
        self._activate_dataset(result_name, aggregated_df, emit_datasets=True)
        self.status_changed.emit(f"Created aggregated dataset: {result_name}")

    def _on_aggregate_error(self, message: str) -> None:
        self.aggregate_button_enabled.emit(True)
        self.error_occurred.emit("Aggregate error", message)

    def _on_merge_finished(self, result_name: str, merged_df) -> None:
        self.merge_button_enabled.emit(True)
        self.state.dataframes[result_name] = merged_df
        self._activate_dataset(result_name, merged_df, emit_datasets=True)
        self.status_changed.emit(f"Created merged dataset: {result_name}")

    def _on_merge_error(self, message: str) -> None:
        self.merge_button_enabled.emit(True)
        self.error_occurred.emit("Merge error", message)

    def _build_result_name(self, left_name: str, right_name: str, join_type: str) -> str:
        base_name = f"Merged {join_type.title()} {Path(left_name).stem} + {Path(right_name).stem}"
        return self._build_unique_name(base_name, self.state.dataframes)

    def _build_aggregate_result_name(self, dataset_name: str, output_name: str) -> str:
        base_name = output_name.strip() or f"Aggregated {Path(dataset_name).stem}"
        return self._build_unique_name(base_name, self.state.dataframes)

    def _build_prediction_result_name(self) -> str:
        active_name = self.state.active_dataset_name or "Dataset"
        base_name = f"Predicted {Path(active_name).stem}"
        return self._build_unique_name(base_name, self.state.dataframes)

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
            return ""
        summary = [
            self.state.selected_model_name or "Model",
            f"Target: {self.state.target_column or '-'}",
            f"Features: {len(self.state.feature_columns)}",
        ]
        if "accuracy" in self.state.metrics:
            summary.append(f"Accuracy: {self.state.metrics['accuracy']:.4f}")
        if self.state.incremental_model_bundle is not None:
            summary.append("Incremental Ready")
        return "\n".join(summary)

    def _emit_trained_models(self) -> None:
        model_names = sorted(self.state.trained_models.keys(), key=str.casefold)
        self.trained_models_changed.emit(model_names, self.state.active_model_name or "")
        self.trained_model_changed.emit(self.describe_trained_model())

    def _clear_active_model_state(self) -> None:
        self.state.trained_model = None
        self.state.metrics.clear()
        self.state.feature_importance = None
        self.state.feature_columns = []
        self.state.target_column = None
        self.state.incremental_model_bundle = None
        self.state.incremental_features = []
        self.state.incremental_target = None
        self.state.incremental_model_name = None
        self.state.incremental_task_type = None
        self.trained_model_changed.emit("")

    def _get_model_entry(self, model_name: str | None) -> dict | None:
        if model_name:
            return self.state.trained_models.get(model_name)
        if self.state.active_model_name:
            return self.state.trained_models.get(self.state.active_model_name)
        return None

    def _build_model_entry_name(self, base_name: str) -> str:
        return self._build_unique_name(base_name.strip() or "Model", self.state.trained_models)

    def _build_range_dataset_name(self, dataset_name: str, time_column: str, start_text: str, end_text: str) -> str:
        base_name = dataset_name.strip() or self._default_range_dataset_name(time_column, start_text, end_text)
        return self._build_unique_name(base_name, self.state.dataframes)

    def _build_metrics_text(self, metrics: dict[str, float]) -> str:
        if not metrics:
            return ""
        metric_lines = [f"{key}: {value:.4f}" for key, value in metrics.items()]
        return "Training complete\n" + "\n".join(metric_lines)

    def _build_unique_name(self, base_name: str, existing: dict) -> str:
        candidate = base_name
        suffix = 2
        while candidate in existing:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _default_range_dataset_name(self, time_column: str, start_text: str, end_text: str) -> str:
        active_name = self.state.active_dataset_name or "Dataset"
        start_part = start_text.strip() or "Start"
        end_part = end_text.strip() or "End"
        return f"{Path(active_name).stem} {time_column} {start_part} To {end_part}"

    def _activate_dataset(self, dataset_name: str, df, emit_datasets: bool = False) -> None:
        self.state.active_dataset_name = dataset_name
        self._set_working_df(df)
        if emit_datasets:
            self._emit_datasets()

    def _set_working_df(self, df, status_message: str | None = None) -> None:
        updated_df = df.copy()
        self.state.working_df = updated_df
        if self.state.active_dataset_name is not None:
            # Keep the selected loaded dataset in sync with Clean edits.
            self.state.dataframes[self.state.active_dataset_name] = updated_df.copy()
        self._emit_working_df()
        if status_message:
            self.status_changed.emit(status_message)

    def _clear_train_thread(self) -> None:
        self.train_worker = None
        self.train_thread = None

    def _clear_aggregate_thread(self) -> None:
        self.aggregate_worker = None
        self.aggregate_thread = None

    def _clear_merge_thread(self) -> None:
        self.merge_worker = None
        self.merge_thread = None
