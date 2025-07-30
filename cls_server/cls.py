from typing import Any

class Cls:
    def __init__(self):
        self.model = None
        self.class_priors = None
        self.features = None

    def load_model(self, model_dict: dict):
        try:
            self.model = model_dict
            # נניח ש־model_dict כולל מפתחות אלה:
            self.class_priors = model_dict.get("class_priors")
            self.features = model_dict.get("features")
            if self.class_priors is None or self.features is None:
                raise ValueError("Model dict missing required keys")
        except Exception as e:
            raise ValueError(f"Failed to load model from dictionary: {e}")

    def predict(self, row: dict) -> Any:
        if not self.model:
            raise ValueError("No model loaded")
        res = {}
        for label in self.model.keys():
            prob = self.class_priors[label]
            for feature in self.features:
                val_input = row.get(feature)
                prob *= self.model[label][feature].get(val_input, 1e-6)
            res[label] = prob
        # מחזיר את התווית עם ההסתברות הכי גבוהה
        return max(res, key=res.get)

