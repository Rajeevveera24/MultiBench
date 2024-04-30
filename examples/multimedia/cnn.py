import torch

class CNN(torch.nn.Module):
    
    def __init__(self, in_channels, outdim):
        super(CNN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=3, stride=2)
        self.conv_2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv_3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv_4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv_5 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(7168, outdim)
        self.dropout_1 = torch.nn.Dropout(0.2)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.activation_1 = torch.nn.ReLU()
        self.activation_2 = torch.nn.ReLU()
        self.activation_3 = torch.nn.ReLU()
        self.activation_4 = torch.nn.ReLU()
        self.activation_5 = torch.nn.ReLU()
        self.activation_6 = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.activation_2(x)
        x = self.dropout_1(x)
        x = self.conv_3(x)
        x = self.activation_3(x)
        x = self.conv_4(x)
        x = self.activation_4(x)
        x = self.dropout_2(x)
        x = self.conv_5(x)
        x = self.activation_5(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation_6(x)
        return x