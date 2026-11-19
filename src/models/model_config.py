from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.data.preprocess import make_preprocessor
from src.utils.utilities import SEED


def model_configs(quick=False):
    specs = {
        "majority_baseline": (DummyClassifier(strategy="most_frequent"), False, {}),
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=SEED), True,
                                {"model__C": [0.1, 1, 10]}),
        "rbf_svm": (SVC(kernel="rbf"), True, {"model__C": [1, 10], "model__gamma": ["scale", 0.1]}),
        "gaussian_nb": (GaussianNB(), False, {"model__var_smoothing": [1e-9, 1e-7]}),
        "decision_tree": (DecisionTreeClassifier(random_state=SEED), False,
                          {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [1, 5]}),
        "random_forest": (RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1), False,
                          {"model__max_depth": [6, None], "model__min_samples_leaf": [1, 3]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=SEED), False,
                              {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]}),
    }
    for name, (model, scale, grid) in specs.items():
        if quick:
            grid = {key: values[:1] for key, values in grid.items()}
        yield name, Pipeline([("preprocess", make_preprocessor(scale)), ("model", model)]), grid
