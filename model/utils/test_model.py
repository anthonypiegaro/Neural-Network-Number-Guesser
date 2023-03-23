import numpy as np
from sklearn.metrics import accuracy_score

def test_model(model, feature_data, target_data):
    predictions = np.argmax(model.predict(feature_data), axis=-1)
    score = accuracy_score(target_data, predictions)
    return score
