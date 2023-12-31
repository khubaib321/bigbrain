import random as _random
import decimal as _decimal

import nn as _nn

_INPUT_SIZE = 784  # 28x28 picture
_TARGET_OUTPUT = [
    _decimal.Decimal(27.57),
    _decimal.Decimal(53.93),
    _decimal.Decimal(89.15),
]


def main():
    inputs = [_decimal.Decimal(_random.randint(0, 99)) for _ in range(_INPUT_SIZE)]

    nn = _nn.NeuralNetwork(target=_TARGET_OUTPUT)
    nn.run(inputs=inputs)


if __name__ == "__main__":
    main()
