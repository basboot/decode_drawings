from global_settings import DRAWING_VOLUME
from helper_functions import butter_lowpass_filter
from neural_network import NeuralNetwork
from plot_data import plot_data_grid
from process_video import get_video_data
import numpy as np
import torch
import os


# Check if GPU is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("MPS device is available. Tensor on MPS:", x)
else:
    print("MPS device not found.")

print("PyTorch version:", torch.__version__)

# Determine device
device = mps_device if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

VIDEO = "2" # video to predict on
MODEL = "1"  # video the model was trained on


# Determine device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")


ball_information, video_information = get_video_data(VIDEO)
ball_information = np.array(ball_information)

# Adjust angles in ball_information to be within [-90, 90]
ball_information[:, :, 4] = (ball_information[:, :, 4] + 180) % 180
ball_information[:, :, 4] = np.where(ball_information[:, :, 4] > 90, ball_information[:, :, 4] - 180, ball_information[:, :, 4])


# Flatten ball_information: 3x5 => 15
ball_information_flat = ball_information.reshape(ball_information.shape[0], -1)

# norm
ball_information_flat += [-1269/2, -719/2, -105, -105, 0] * 3
ball_information_flat /= [1269/2, 719/2, 105, 105, 90] * 3


# Convert to np arrays to pt tensor, and put on gpu
X_predict_tensor = torch.tensor(ball_information_flat, dtype=torch.float32).to(device)



input_size = ball_information_flat.shape[1] 
output_size = 3  # x, y, z

loaded_model = NeuralNetwork(input_size, output_size)

# load model parameters
model_filename = f"model_{MODEL}.pth"
model_path = os.path.join("models", model_filename)

if os.path.exists(model_path):
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))  
    loaded_model.to(device)
    loaded_model.eval() # set model to evaluation mode for predictions
    print(f"Model loaded successfully from {model_path}")
else:
    print(f"Error: Model file not found at {model_path}")
    exit()


with torch.no_grad(): # gradient not needed for eval
    predictions_tensor = loaded_model(X_predict_tensor)
    coords = predictions_tensor.cpu().numpy() 

    # de-norm
    coords *= np.array([11, 1, 11])
    coords -= np.array([0, 0, -18])


# filter audio
video_information['volume'] = butter_lowpass_filter(video_information['volume'], 1, 60, 2)

drawing_x, drawing_y = [], []

drawing_x, drawing_y = [], []
for i in range(len(coords)):
    if i > len(video_information["volume"]) - 1:
        break
    if video_information["volume"][i] > DRAWING_VOLUME:
        drawing_x.append(coords[i, 0])
        drawing_y.append(coords[i, 2])

# Define configurations for each plot
plot_configs = [
    {
        "x_data": coords[:, 0], "y_data": coords[:, 2],
        "xlabel": 'X', "ylabel": 'Z', "title": 'Top View (X-Z)',
        "marker": 'o', "marker_size": 25, "color": None, "equal_axis": True, "mirror_y": True
    },
    {
            "x_data": drawing_x, "y_data": drawing_y,
            "xlabel": 'X', "ylabel": 'Z', "title": 'Drawing',
            "marker": 'o', "marker_size": 1, "color": "k",  "equal_axis": True, "scatter_only": True, "mirror_y": True
        },
]

# plot data
plot_data_grid(plot_configs, len(coords), VIDEO, show_plot=True, output_dir="output_ai")

with open(f"drawings_ai/drawing{VIDEO}.txt", "w") as f:
    for x, z in zip(drawing_x, drawing_y):
        f.write(f"{x} {z}\n")








