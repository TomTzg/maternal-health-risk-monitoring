import joblib
from src.data.preprocess import validate_frame
from src.utils.utilities import ARTIFACTS, FEATURES


def predict(frame, model_path=None):
    model = joblib.load(model_path or ARTIFACTS / "best_model.joblib")
    clean = validate_frame(frame, require_target=False)
    result = frame.copy()
    result["PredictedRisk"] = model.predict(clean[FEATURES])
    return result
