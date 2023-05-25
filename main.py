import nn as _nn
import random as _random
import decimal as _decimal


_INPUTS_SIZE = 1000


def main():
    bias = _decimal.Decimal(0)  # Y-intercept of the output graph.
    inputs = [_decimal.Decimal(_random.randint(0, 99)) for _ in range(_INPUTS_SIZE)]
    target_output = _decimal.Decimal(7)

    nn = _nn.NeuralNetwork(bias=bias, learning_rate=_decimal.Decimal(0.05))
    nn.run(inputs=inputs, target_output=target_output)

    print(f"\nNN Output: {nn.output}\n")


if __name__ == "__main__":
    main()
