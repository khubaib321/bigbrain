import nn as _nn
import random as _random
import decimal as _decimal


_INPUTS_SIZE = 100000


if __name__ == "__main__":
    inputs = []
    bias = _decimal.Decimal()
    target_output = _decimal.Decimal(0.7)

    for _ in range(_INPUTS_SIZE):
        value = _random.random()
        weight = _random.random()

        # TODO: I think input values should sourced from a file or an external data source.
        _nn.Input(value=value, weight=weight)

    nn = _nn.NeuralNetwork(bias=bias)
    nn.run(input_vector=inputs, target_output=target_output)

    print(f"\nNN Output: {nn.output}\n")
