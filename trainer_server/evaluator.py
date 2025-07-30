import pandas as pd

class Evaluator:
    def __init__(self,classifier):
        self.classifier = classifier

    def evaluate_model(self,X_test: pd.DataFrame,y_test: pd.Series):
        correct = 0
        for i, row in X_test.iterrows() :
            prediction = self.classifier.predict(row)
            if prediction == y_test.loc[i] :
                correct += 1
        accuracy = correct / len(y_test) * 100

        return float(accuracy)