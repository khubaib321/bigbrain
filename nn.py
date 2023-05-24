import copy as _copy
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
            (input.computed_value for input in self._input_vector),
            start=_decimal.Decimal(),
        )

    def _activation_function(self) -> _decimal.Decimal:
        """
        To introduce non-linearity to the output, use one the common activation functions.
        Activation function is applied to the calculated linear output. Commonly used functions:
        - Sigmoid Function
        - ReLU (Rectified Linear Unit) Function
        - Tanh (Hyperbolic Tangent) Function
        - Softmax Function
        """

        if self._linear_output is None:
            raise ValueError(
                "Linear output must be calculated before applying activation function"
            )

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

        self._loss = None
        self._output = None
        self._hidden_layers = []
        self._computed_layers = []

    def _compute_layer(self, layer: list["Neuron"]) -> list[Input]:
        output_layer = [
            Input(value=neuron.output, weight=_decimal.Decimal(0.05))
            for neuron in layer
        ]
        self._computed_layers.append(layer)

        return output_layer

    @property
    def loss(self) -> _decimal.Decimal:
        if self._loss is None:
            raise ValueError("Run the neural network using the `compute_output` method")

        return self._loss

    @property
    def output(self) -> _decimal.Decimal:
        if self._output is None:
            raise ValueError("Run the neural network using the `compute_output` method")

        return self._output

    def _compute_loss(self, target_ouput: _decimal.Decimal):
        """
        Loss function for the neural network. 
        Computes Squared Loss of the network's output.
        This loss function is only valid for running the NN on a single input (or input vector).
        If NN is running on multiple inputs (or input vectors) then MSE can/should be used a loss function.
        """

        # I have yet to learn what it means to run a single input vs multiple inputs.
        self._loss = (self.output - target_ouput) ** 2

    def _adjust_weights(self, input_vector: list[Input]) -> list[Input]:
        """
        Adjust weights according to the latest loss values.
        """

        loss_value = self.loss
        weight_adjusted_iv: list[Input] = []

        for _input in input_vector:
            # TODO: Use Back Propagation to adjust the weight according to loss value.
            # TODO: Define Learning Rate to control the degree of weight adjustments.
            weight_adjusted_iv.append(_input)

        return weight_adjusted_iv

    def compute_output(
        self, input_vector: list[Input], at_depth: int = 0
    ) -> None:
        """
        Runs the neural network on the input vector.
            Width = 3
            Depth = 2
            Neuron = *

            Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
                *       ->      *         ->      *         ->      *       \
                *       ->      *         ->      *         ->      *       ->      * (output neuron)
                *       ->      *         ->      *         ->      *       /
        """

        # TODO: Use iterative approach. It is better than recursive approach.

        while at_depth < self._depth:
            input_layer = [
                Neuron(input_vector=input_vector, bias=self._bias)
                for _ in range(self._width)
            ]

            next_input_vector = self._compute_layer(input_layer)

            return self.compute_output(
                input_vector=next_input_vector, at_depth=at_depth + 1
            )

        # TODO: Output layer typically uses different activation function than the hidden layers (why?).
        # Adjust the Neuron interface to allow use of custom activation functions.
        self._output = Neuron(input_vector=input_vector, bias=self._bias).output

    def run(
        self, *, input_vector: list[Input], target_output: _decimal.Decimal
    ) -> None:
        min_loss = _decimal.Decimal(0.05)
        current_iv = _copy.deepcopy(input_vector)

        # TODO: Verify: Loop will not run indefinitely, right?
        while self.loss > min_loss:
            print("\nCompute Goal: Minimize loss")

            self.compute_output(current_iv)
            self._compute_loss(target_output)
            current_iv = self._adjust_weights(input_vector)

            print(f"Current loss value: {self.loss}")
            print(f"Min acceptable loss value: {min_loss}")
