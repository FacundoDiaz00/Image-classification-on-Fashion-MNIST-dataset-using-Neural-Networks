from torch import nn

"""
    Se generan varias arquitecturas de prueba que no se borran con el fin de poder mostrar las modificaciones que nos llevaron
    a la elección de las tres arquitecturas finales.
    
    Se elige la Red 2, Red 4 y Red 7.
"""


class Red2(nn.Module):
    """
        Segunda arquitectura propuesta: tiene una única capa oculta de 512 unidades con una función de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class Red3(nn.Module):
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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class Red4(nn.Module):
    """
    Cuarta arquitectura propuesta: tiene tres capas ocultas de 512, 256 y 128 unidades respectivamente, todas con una
    función de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class Red5(nn.Module):
    """
        Sexta arquitectura propuesta: tiene cuatro capas ocultas de 512, 256, 128 y 64 unidades respectivamente, todas con una
        función de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class Red6(nn.Module):
    """
     Séptima arquitectura propuesta: tiene tres capas ocultas de 256, 512 y 256 unidades respectivamente, todas con una
     función de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class Red7(nn.Module):
    """
        Primera arquitectura propuesta: tiene tres capas ocultas de 256, 128 y 64 unidades respectivamente, todas con una
        función de activación ReLU.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_linear_stack = nn.Sequential(

            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits


class RedMuchasCapas(nn.Module):
    """
     Arquitectura de prueba: tiene {num_hidden_layers} capas con {hidden_layer_size} unidades cada una. Para experimentar
     intentamos usar 24 capas ocultas con 20 unidades cada una, todas con función de activación ReLU.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        num_hidden_layers = 24
        hidden_layer_size = 20

        layers = [nn.Linear(28 * 28, hidden_layer_size), nn.ReLU()]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layer_size, 10))

        self.linear_linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_linear_stack(x)
        return logits