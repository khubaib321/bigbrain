import decimal as _decimal
import random as _random

import nn as _nn

_INPUT_SIZE = 10000
_TARGET_OUTPUT = [
    _decimal.Decimal("27.57"),
    _decimal.Decimal("53.93"),
    _decimal.Decimal("89.15")
]


def main():
    bias = _decimal.Decimal(0)  # Y-intercept of the output graph.
    inputs = [_decimal.Decimal(_random.randint(0, 99)) for _ in range(_INPUT_SIZE)]

    nn = _nn.NeuralNetwork(
        bias=bias, 
        target=_TARGET_OUTPUT
    )
    nn.run(inputs=inputs)


if __name__ == "__main__":
    main()
