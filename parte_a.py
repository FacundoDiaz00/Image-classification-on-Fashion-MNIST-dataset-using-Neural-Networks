from torch import nn


class Red1(nn.Module):
    """Red pedido en la parte A: tiene una capa oculta de 32 unidades con una activación Sigmoide. No tiene una función
    de activación en la salida dado que se va a entrenar con la función de pérdida de entropía cruzada.
    """""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 32),
            nn.Sigmoid(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits