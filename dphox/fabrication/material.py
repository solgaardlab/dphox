from typing import Tuple


class Material:
    def __init__(self, name: str, facecolor: Tuple[float, float, float]=None, epsilon: float=None):
        self.name = name
        self.epsilon = epsilon
        self.facecolor = facecolor

    def __str__(self):
        return self.name


SILICON = Material('Silicon', (0.3, 0.3, 0.3), 12.6)
POLYSILICON = Material('Poly-Si', (0.5, 0.5, 0.5), 12.6)
OXIDE = Material('Oxide', (0.6, 0, 0), 2.085)
NITRIDE = Material('Nitride', (0, 0, 0.7), 3.985)
LS_NITRIDE = Material('Low-Stress Nitride', (0, 0.4, 1))
LT_OXIDE = Material('Low-Temp Oxide', (0.8, 0.2, 0.2), 2.085)
ALUMINUM = Material('Aluminum', (0, 0.5, 0))
ALUMINA = Material('Alumina', (0.2, 0, 0.2), 1.75)
ETCH = Material('Etch', (0, 0, 0))
