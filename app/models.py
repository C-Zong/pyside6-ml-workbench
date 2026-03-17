from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR


def get_model(model_name: str, task_type: str):
    if task_type == "classification":
        mapping = {
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "SVC": SVC(probability=True),
        }
    else:
        mapping = {
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
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
        ]
    return [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "LinearRegression",
        "SVR",
    ]