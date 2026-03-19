from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR


INCREMENTAL_MODEL_NAMES = {"SGDClassifier", "SGDRegressor"}


def get_model(model_name: str, task_type: str, model_params: dict | None = None):
    params = model_params or {}
    if task_type == "classification":
        mapping = {
            "RandomForestClassifier": lambda: RandomForestClassifier(random_state=42, **params),
            "GradientBoostingClassifier": lambda: GradientBoostingClassifier(random_state=42, **params),
            "LogisticRegression": lambda: LogisticRegression(max_iter=2000, **params),
            "SVC": lambda: SVC(probability=True, **params),
            "SGDClassifier": lambda: SGDClassifier(random_state=42, **params),
        }
    else:
        mapping = {
            "RandomForestRegressor": lambda: RandomForestRegressor(random_state=42, **params),
            "GradientBoostingRegressor": lambda: GradientBoostingRegressor(random_state=42, **params),
            "SGDRegressor": lambda: SGDRegressor(random_state=42, **params),
            "SVR": lambda: SVR(**params),
        }
    return mapping[model_name]()


def get_model_options(task_type: str):
    if task_type == "classification":
        return [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
            "SVC",
            "SGDClassifier",
        ]
    return [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR",
        "SGDRegressor",
    ]


def is_incremental_model(model_name: str) -> bool:
    return model_name in INCREMENTAL_MODEL_NAMES
