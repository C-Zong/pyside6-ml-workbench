from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class AppState:
    """Store loaded datasets, the active working dataframe, and training history."""

    # Loaded source tables keyed by displayed dataset name.
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Name of the dataset currently selected in the shared dataset panel.
    active_dataset_name: Optional[str] = None
    # Mutable dataframe used by clean/process/train workflows.
    working_df: Optional[pd.DataFrame] = None
    # Column currently selected in the clean workflow.
    clean_column: Optional[str] = None
    # Columns selected as model input features.
    feature_columns: List[str] = field(default_factory=list)
    # Column selected as the prediction target.
    target_column: Optional[str] = None
    # Currently selected model name in the UI.
    selected_model_name: str = "RandomForestClassifier"
    # Inferred ML task type.
    task_type: str = "classification"
    # Evaluation metrics from the latest training run.
    metrics: Dict[str, float] = field(default_factory=dict)
    # Feature-importance table produced by the trained model.
    feature_importance: Optional[pd.DataFrame] = None
    # Loaded and trained model entries keyed by display name.
    trained_models: Dict[str, dict] = field(default_factory=dict)
    # Name of the model currently selected in the left panel.
    active_model_name: Optional[str] = None
    # Trained pipeline or estimator object.
    trained_model: Optional[object] = None
    # Stored incremental model state for true continuation.
    incremental_model_bundle: Optional[object] = None
    # Feature set used by the stored incremental model.
    incremental_features: List[str] = field(default_factory=list)
    # Target used by the stored incremental model.
    incremental_target: Optional[str] = None
    # Model name used by the stored incremental model.
    incremental_model_name: Optional[str] = None
    # Task type used by the stored incremental model.
    incremental_task_type: Optional[str] = None
