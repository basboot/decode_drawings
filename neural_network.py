import torch

# TODO: tune NN (try smaller network icm large dataset)
# TODO: normalize input/output
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu3 = torch.nn.ReLU()  
        self.fc4 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        x = self.relu3(x)  
        x = self.fc4(x)
        return x
