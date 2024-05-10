import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=0) # Applies 128 filters of size 9x9 (kernel_size=9) to the input image.
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.iden = nn.Identity()
        self.init_weights()  # Initialize weights

    def init_weights(self):
      # Initialize weights using He initialization
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x) #Patch Extraction and Representation with 1st layer using convolutional filters (nn.Conv2d).
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x) #Convolutional Layer 2: Non-linear Mapping -> maps the extracted features to a higher-dimensional space.
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x) #Convolutional Layer 3: Reconstruction of the high-resolution image from the mapped features.
        x = self.iden(x)

        # Apply interpolation to resize the output to match the target label size (20x20)
        x = nn.functional.interpolate(x, size=(20, 20), mode='bilinear', align_corners=False)

        return x