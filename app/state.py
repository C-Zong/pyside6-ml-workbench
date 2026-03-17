from dataclasses import dataclass, field
from typing import Optional, List, Dict
import pandas as pd


@dataclass
class AppState:
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    merged_df: Optional[pd.DataFrame] = None
    cleaned_df: Optional[pd.DataFrame] = None
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    selected_model_name: str = "RandomForestClassifier"
    task_type: str = "classification"  # classification / regression
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[pd.DataFrame] = None
    trained_model: Optional[object] = None