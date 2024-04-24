import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=0)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

"""
def model():
    SRCNN = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(9, 9), stride=1, padding=0, bias=True),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(5, 5), stride=1, padding=0, bias=True)
    )

    optimizer = optim.Adam(SRCNN.parameters(), lr=0.0003)
    loss_fn = nn.MSELoss()

    return SRCNN, optimizer, loss_fn
"""
