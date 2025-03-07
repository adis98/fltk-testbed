import torch
from torch import nn
from fltk.nets.speclayers.layer import compute_conv_output_size


class Cifar100LeNet(nn.Module):
    def __init__(self, inputsize, n_outputs=100, nc_per_task=5, window="sliding"):
        super().__init__()

        self.nc_per_task = nc_per_task
        self.n_outputs = n_outputs
        self.window = window
        ncha, size, _ = inputsize
        self.conv1 = nn.Conv2d(in_channels=ncha, out_channels=20, kernel_size=5, padding=2, stride=1)
        s = compute_conv_output_size(size, 5, stride=1, padding=2)  # 32
        self.pool1 = torch.nn.MaxPool2d(3, stride=2, padding=1)  # 16
        s = compute_conv_output_size(s, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, stride=1)  # 16
        s = compute_conv_output_size(s, 5, stride=1, padding=2)
        self.pool2 = torch.nn.MaxPool2d(3, stride=2, padding=1)  # 8
        s = compute_conv_output_size(s, 3, padding=1, stride=2)  # 32

        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        self.fc1 = nn.Linear(s * s * 50, 800)
        self.fc2 = nn.Linear(800, 500)
        self.last = torch.nn.Linear(500, self.n_outputs)
        self.avg_neg = []
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.layer_keys = [['conv1'], ['conv2'], ['fc1'], ['fc2'], ['last']]

    def forward(self, x, t):
        if x.size(1) != 3:
            bsz = x.size(0)
            x = x.view(bsz, 3, 32, 32)
        act1 = self.relu(self.conv1(x))

        h = self.drop1(self.pool1(act1))

        act2 = self.relu(self.conv2(h))

        h = self.drop1(self.pool2(act2))

        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        act8 = self.relu(self.fc2(h))
        output = self.last(act8)
        # make sure we predict classes within the current task
        offset1 = 0 if self.window != "sliding" else int(t * self.nc_per_task)
        # offset1 = int(t * self.nc_per_task)
        offset2 = self.n_outputs if self.window == "full" else int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
