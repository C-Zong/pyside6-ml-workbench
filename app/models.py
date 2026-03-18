from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR


INCREMENTAL_MODEL_NAMES = {"SGDClassifier", "SGDRegressor"}


def get_model(model_name: str, task_type: str):
    if task_type == "classification":
        mapping = {
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "SVC": SVC(probability=True),
            "SGDClassifier": SGDClassifier(random_state=42),
        }
    else:
        mapping = {
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "SGDRegressor": SGDRegressor(random_state=42),
            "SVR": SVR(),
        }
    return mapping[model_name]


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
