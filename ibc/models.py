from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=30):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 4**3),
            nn.ReLU(),
            nn.Linear(4**3, 4**5),
            nn.ReLU(),
            nn.Linear(4**5, 4**7),
            nn.ReLU(),
            nn.Linear(4**7, 4**3),
            nn.ReLU(),
            nn.Linear(4**3, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
