from neural_network import NeuralNetwork
from process_video import get_video_data
import numpy as np


import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split 
import os 


TEST_PCT = 0.01 # use almost everything for training

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


# TODO: create option to combine multiple datasets

VIDEO = "1"

# 22 is slechter dan 21

ball_information, video_information, location_information = get_video_data(VIDEO, createTrainingData=True)

# ball_information2, video_information2, location_information2 = get_video_data("22", createTrainingData=True)

ball_information = np.array(ball_information)
# ball_information2 = np.array(ball_information2)
location_information = np.array(location_information)
# location_information2 = np.array(location_information2)

# Combine ball and video information from both datasets
# ball_information = np.concatenate((ball_information, ball_information2), axis=0)
# location_information = np.concatenate((location_information, location_information2), axis=0)

# Adjust angles in ball_information to be within [-90, 90]
ball_information[:, :, 4] = (ball_information[:, :, 4] + 180) % 180
ball_information[:, :, 4] = np.where(ball_information[:, :, 4] > 90, ball_information[:, :, 4] - 180, ball_information[:, :, 4])

# Flatten ball_information: 3x5 => 15
ball_information_flat = ball_information.reshape(ball_information.shape[0], -1)
# Location already flat: 3
location_information = np.array(location_information)


# add noise
# Add different noise levels to different columns/features
noise_std = [5, 5, 8, 8, 10] * 3  # Repeat for 3 ball detections
noise = np.random.normal(0, noise_std, ball_information_flat.shape)
ball_information_flat_noisy = ball_information_flat + noise

# in: x, y, w, h, angle
# out: x, y, z 18 +- 11, 0 - 1, 18 +- 11

ball_information_flat += [-1269/2, -719/2, -105, -105, 0] * 3
ball_information_flat /= [1269/2, 719/2, 105, 105, 90] * 3
location_information += np.array([0, 0, -18]) 
location_information /= np.array([11, 1, 11])


# Convert to np arrays to pt tensors
X_tensor = torch.tensor(ball_information_flat, dtype=torch.float32) 
y_tensor = torch.tensor(location_information, dtype=torch.float32) 

# split train and test

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=TEST_PCT, random_state=42)

# move to gpu (if possible)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Create DataLoaders

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 256 # TODO: tune
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle test data

input_size = ball_information_flat.shape[1] 
output_size = location_information.shape[1] 
model = NeuralNetwork(input_size, output_size).to(device)
print(model)

# Loss function and Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train NN
num_epochs = 200 # TODO: tune => experiment with less epochs, loss is 0.0000 after 20 epochs! 15-20 seems enough when normalizing and training with enough data
for epoch in range(num_epochs):
    model.train() # set model to training mode
    epoch_loss = 0
    for batch_X, batch_y in train_dataloader: 
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader):.8f}')

print("Training finished.")

# evaluate
model.eval() # Set model to evaluation mode
test_loss = 0
with torch.no_grad(): 
    for batch_X, batch_y in test_dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_dataloader)
print(f'Average Test MSE: {avg_test_loss:.4f}')

# save model
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

model_filename = f"model_{VIDEO}.pth"
model_path = os.path.join(MODELS_DIR, model_filename)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
