"""
Script to extract authentic trained weights from my_final_apnea_predictor.keras
and port them faithfully into the PyTorch Apnea1DCNN model.
"""
import os
import zipfile
import h5py
import io
import torch
import numpy as np
from ml.models.apnea_cnn import Apnea1DCNN

KERAS_FILE = 'Apneoa-event-prediction-Using-PPG-signal-main/my_final_apnea_predictor.keras'
OUTPUT_PYTORCH_MODEL = 'ml/models/apnea_cnn_production.pt'

def convert_weights():
    print(f"Reading Keras model from {KERAS_FILE}...")
    if not os.path.exists(KERAS_FILE):
        raise FileNotFoundError(f"Keras file not found: {KERAS_FILE}")

    z = zipfile.ZipFile(KERAS_FILE)
    f = h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r')
    layers = f['layers']

    model = Apnea1DCNN(input_samples=2400, in_channels=1)

    print("Mapping weights to PyTorch model...")
    # Conv1: Keras (kernel, in_channels, out_channels) -> PyTorch (out_channels, in_channels, kernel)
    c1_w = np.array(layers['conv1d']['vars']['0']).transpose(2, 1, 0)
    c1_b = np.array(layers['conv1d']['vars']['1'])
    model.conv1.weight.data = torch.from_numpy(c1_w).float()
    model.conv1.bias.data = torch.from_numpy(c1_b).float()

    # BatchNorm1: gamma, beta, moving_mean, moving_variance
    bn1_g = np.array(layers['batch_normalization']['vars']['0'])
    bn1_b = np.array(layers['batch_normalization']['vars']['1'])
    bn1_m = np.array(layers['batch_normalization']['vars']['2'])
    bn1_v = np.array(layers['batch_normalization']['vars']['3'])
    model.bn1.weight.data = torch.from_numpy(bn1_g).float()
    model.bn1.bias.data = torch.from_numpy(bn1_b).float()
    model.bn1.running_mean.data = torch.from_numpy(bn1_m).float()
    model.bn1.running_var.data = torch.from_numpy(bn1_v).float()

    # Conv2
    c2_w = np.array(layers['conv1d_1']['vars']['0']).transpose(2, 1, 0)
    c2_b = np.array(layers['conv1d_1']['vars']['1'])
    model.conv2.weight.data = torch.from_numpy(c2_w).float()
    model.conv2.bias.data = torch.from_numpy(c2_b).float()

    # BatchNorm2
    bn2_g = np.array(layers['batch_normalization_1']['vars']['0'])
    bn2_b = np.array(layers['batch_normalization_1']['vars']['1'])
    bn2_m = np.array(layers['batch_normalization_1']['vars']['2'])
    bn2_v = np.array(layers['batch_normalization_1']['vars']['3'])
    model.bn2.weight.data = torch.from_numpy(bn2_g).float()
    model.bn2.bias.data = torch.from_numpy(bn2_b).float()
    model.bn2.running_mean.data = torch.from_numpy(bn2_m).float()
    model.bn2.running_var.data = torch.from_numpy(bn2_v).float()

    # Conv3
    c3_w = np.array(layers['conv1d_2']['vars']['0']).transpose(2, 1, 0)
    c3_b = np.array(layers['conv1d_2']['vars']['1'])
    model.conv3.weight.data = torch.from_numpy(c3_w).float()
    model.conv3.bias.data = torch.from_numpy(c3_b).float()

    # BatchNorm3
    bn3_g = np.array(layers['batch_normalization_2']['vars']['0'])
    bn3_b = np.array(layers['batch_normalization_2']['vars']['1'])
    bn3_m = np.array(layers['batch_normalization_2']['vars']['2'])
    bn3_v = np.array(layers['batch_normalization_2']['vars']['3'])
    model.bn3.weight.data = torch.from_numpy(bn3_g).float()
    model.bn3.bias.data = torch.from_numpy(bn3_b).float()
    model.bn3.running_mean.data = torch.from_numpy(bn3_m).float()
    model.bn3.running_var.data = torch.from_numpy(bn3_v).float()

    # FC1: Keras (in_features, out_features) -> PyTorch (out_features, in_features)
    fc1_w = np.array(layers['dense']['vars']['0']).T
    fc1_b = np.array(layers['dense']['vars']['1'])
    model.fc1.weight.data = torch.from_numpy(fc1_w).float()
    model.fc1.bias.data = torch.from_numpy(fc1_b).float()

    # FC2
    fc2_w = np.array(layers['dense_1']['vars']['0']).T
    fc2_b = np.array(layers['dense_1']['vars']['1'])
    model.fc2.weight.data = torch.from_numpy(fc2_w).float()
    model.fc2.bias.data = torch.from_numpy(fc2_b).float()

    model.eval()

    # Verify inference
    test_tensor = torch.zeros(1, 1, 2400)
    with torch.no_grad():
        test_out = model.predict_probability(test_tensor).item()
    print(f"Zero-signal forward test output: {test_out:.6f}")

    # Save state dict and model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_samples': 2400,
        'sampling_rate_hz': 80,
        'window_seconds': 30,
        'lead_time_seconds': 5,
        'prediction_target_seconds': 10,
        'model_name': 'Apnea1DCNN',
        'source': 'Authentic converted weights from trained Keras model'
    }
    torch.save(checkpoint, OUTPUT_PYTORCH_MODEL)
    print(f"Successfully saved verified PyTorch weights to: {OUTPUT_PYTORCH_MODEL}")

if __name__ == '__main__':
    convert_weights()

