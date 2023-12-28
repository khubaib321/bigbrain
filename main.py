import decimal as _decimal
import random as _random

import nn as _nn

_INPUT_SIZE = 10000
_TARGET_OUTPUT = [
    _decimal.Decimal("27.5"),
    _decimal.Decimal("53.9"),
    _decimal.Decimal("89.1")
]


def main():
    bias = _decimal.Decimal(0)  # Y-intercept of the output graph.
    inputs = [_decimal.Decimal(_random.randint(0, 99)) for _ in range(_INPUT_SIZE)]
    target_output = _decimal.Decimal(7)

    nn = _nn.NeuralNetwork(
        bias=bias, 
        target=_TARGET_OUTPUT
    )
    nn.run(inputs=inputs)

    print(f"\nNN Output: {nn.output}\n")


if __name__ == "__main__":
    main()
