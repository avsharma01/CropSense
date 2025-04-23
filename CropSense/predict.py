import numpy as np
import pickle
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("model/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

test_input = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])  

num_features = scaler.n_features_in_
if test_input.shape[1] != num_features:
    raise ValueError(f"Expected input with {num_features} features, but got {test_input.shape[1]} features.")

test_input_scaled = scaler.transform(test_input)
test_input_pca = pca.transform(test_input_scaled)
predicted_label_index = rf_model.predict(test_input_pca)[0]
predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
print(f"\n Predicted Label: {predicted_label}")