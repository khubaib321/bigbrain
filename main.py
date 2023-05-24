import nn as _nn
import decimal as _decimal


_INPUTS_SIZE = 100000


if __name__ == "__main__":
    # Inputs need to be sourced from a file or external data source.
    inputs = [
        _nn.Input(value=_decimal.Decimal(0.05), weight=_decimal.Decimal(0.05))
        for _ in range(_INPUTS_SIZE)
    ]

    nn_runner = _nn.NeuralNetwork(
        bias=_decimal.Decimal(0.05)
    ).prepare_run()

    output = nn_runner(inputs)

    print(f"NN Output: {output}")
