import typing as _typing
import decimal as _decimal


class Input:
    def __init__(self, value: _decimal.Decimal, weight: _decimal.Decimal):
        self.value = value
        self.weight = weight

    @property
    def computed_value(self) -> _decimal.Decimal:
        return self.value * self.weight


class Neuron:
    def __init__(self, input_vector: list[Input], bias: _decimal.Decimal):
        self._bias = bias
        self._linear_output = None
        self._input_vector = input_vector

    @property
    def _weighted_sum(self) -> _decimal.Decimal:
        return sum(
            (input.computed_value for input in self._input_vector), start=_decimal.Decimal()
        )

    def _activation_function(self) -> _decimal.Decimal:
        """
        To introduce non-linearity to the output, use one the common activation functions.
        Activation function is applied to the calculated linear output.
        - Sigmoid Function
        - ReLU (Rectified Linear Unit) Function
        - Tanh (Hyperbolic Tangent) Function
        - Softmax Function
        """

        if self._linear_output is None:
            raise ValueError("Linear output must be calculated before applying activation function")

        # ReLU (Rectified Linear Unit) Function
        return max(0, self._linear_output)

    @property
    def output(self) -> _decimal.Decimal:
        if self._linear_output is None:
            self._linear_output = self._weighted_sum + self._bias

        return self._activation_function()


class NeuralNetwork:
    def __init__(self, *, bias: _decimal.Decimal, width: int = 3, depth: int = 3):
        self._bias = bias
        self._depth = depth
        self._width = width
        self._hidden_layers = []
        self._computed_layers = []

    def _compute_layer(self, layer: list["Neuron"]) -> list[Input]:
        output_layer = [
            Input(value=neuron.output, weight=_decimal.Decimal(0.05)) for neuron in layer
        ]
        self._computed_layers.append(layer)

        return output_layer

    def prepare_run(self) -> _typing.Callable[[list[Input]], _decimal.Decimal]:
        """Returns a runner function that calculates the output of the neural network."""

        def _runner(_iv: list[Input]) -> _decimal.Decimal:
            """
            Width = 3
            Depth = 2
            Neuron = *

            Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
                *       ->      *         ->      *         ->      *       \
                *       ->      *         ->      *         ->      *       ->      *
                *       ->      *         ->      *         ->      *       /
            """

            input_layer = [Neuron(input_vector=_iv, bias=self._bias) for _ in range(self._width)]
            intermediate_output_layer = self._compute_layer(layer=input_layer)

            for _ in range(self._depth):
                hidden_layer = [
                    Neuron(input_vector=intermediate_output_layer, bias=self._bias)
                    for _ in range(self._width)
                ]
                self._hidden_layers.append(hidden_layer)

                intermediate_output_layer = self._compute_layer(layer=hidden_layer)

            output_neuron = Neuron(input_vector=intermediate_output_layer, bias=self._bias)

            return output_neuron.output

        return _runner
