from torch import nn


class RedDropout(nn.Module):
    """
     Tercera arquitectura propuesta: tiene dos capas ocultas de 512 y 256 unidades respectivamente, todas con una función
     de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class RedDropout2(nn.Module):
    """
     Tercera arquitectura propuesta: tiene dos capas ocultas de 512 y 256 unidades respectivamente, todas con una función
     de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)

        return logits


class Red(nn.Module):
    """
     Mejor arquitectura de parte b: tiene dos capas ocultas de 512 y 256 unidades respectivamente, todas con una función
     de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits