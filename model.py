import torch.nn as nn
import torch.nn.functional as F

import time
torch.manual_seed(1) # set the random seed



class Emoji_Classifier(nn.Module):
      def __init__(self):
          super(Emoji_Classifier, self).__init__()
          self.name = "net"
          self.conv1 = nn.Conv2d(3, 10, 5, 2) #in_channels, out_chanels, kernel_size
          self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
          self.conv2 = nn.Conv2d(10, 26, 5, 2) #in_channels, out_chanels, kernel_size
          self.fc1 = nn.Linear(26 * 13 * 13, 120)
          self.fc2 = nn.Linear(120, 8)

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = x.view(-1, 26 * 13 * 13)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          x = x.squeeze(1) #Flatten to batch size
          return x