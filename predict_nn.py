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
MODEL = "19"  # video the model was trained on


# Determine device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")


ball_information, video_information, location_information = get_video_data(VIDEO, createTrainingData=True)

ball_information = np.array(ball_information)
# Flatten ball_information: 3x5 => 15
ball_information_flat = ball_information.reshape(ball_information.shape[0], -1)


# Convert to np arrays to pt tensor, and put on gpu
X_predict_tensor = torch.tensor(ball_information_flat, dtype=torch.float32).to(device)


# TODO: move to separate file
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x



input_size = ball_information_flat.shape[1] 
output_size = 3  # x, y, z

loaded_model = SimpleNN(input_size, output_size)

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



# Define configurations for each plot
plot_configs = [
    {
        "x_data": coords[:, 0], "y_data": coords[:, 2],
        "xlabel": 'X', "ylabel": 'Z', "title": 'Top View (X-Z)',
        "marker": 'o', "marker_size": 25, "color": None, "equal_axis": True, "mirror_y": True
    }
]

# plot data
plot_data_grid(plot_configs, len(coords), VIDEO, show_plot=True)







