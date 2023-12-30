import math as _math
import copy as _copy
import decimal as _decimal
import random as _random


def _relu(weighted_sum: _decimal.Decimal) -> _decimal.Decimal:
    """
    ReLU (Rectified Linear Unit).
    """

    return max(_decimal.Decimal(0), weighted_sum)


def _relu_dx(relu_output: _decimal.Decimal) -> _decimal.Decimal:
    """
    ReLU (Rectified Linear Unit) function derivative.
    """

    return (
        _decimal.Decimal(1)
        if relu_output > _decimal.Decimal(0)
        else _decimal.Decimal(0)
    )


def _softmax(value: _decimal.Decimal) -> _decimal.Decimal:
    return _decimal.Decimal()


def _softmax_dx(value: _decimal.Decimal) -> _decimal.Decimal:
    return _decimal.Decimal()


class Input:
    def __init__(self, value: _decimal.Decimal, weight: _decimal.Decimal):
        self.value = value
        self.weight = weight

    @property
    def computed_value(self) -> _decimal.Decimal:
        return self.value * self.weight


class Neuron:
    def __init__(
        self,
        *,
        bias: _decimal.Decimal,
        target: _decimal.Decimal,
        inputs: list[_decimal.Decimal] = [],
        weights: list[_decimal.Decimal] = [],
        is_in_output_layer: bool = False,
    ):
        self.bias = bias
        self._target = target

        self._loss: _decimal.Decimal | None = None
        self._error: _decimal.Decimal | None = None
        self._linear_output: _decimal.Decimal | None = None

        # Is this Neuron in the output layer or any of the hidden layers?
        self._is_in_output_layer = is_in_output_layer

        if not weights:
            self._weighted_input = [
                Input(value=v, weight=_decimal.Decimal(_random.random()))
                for v in inputs
            ]
        else:
            if len(inputs) != len(weights):
                raise ValueError(
                    "Number of inputs and weights must be equal. "
                    f"Inputs: {len(inputs)}, Weights: {len(weights)}"
                )

            self._weighted_input = [
                Input(value=v, weight=w) for v, w in zip(inputs, weights)
            ]

    @property
    def input_vector(self) -> list[_decimal.Decimal]:
        return [input.value for input in self._weighted_input]

    @property
    def weight_vector(self) -> list[_decimal.Decimal]:
        return [input.weight for input in self._weighted_input]

    @property
    def weighted_input(self) -> list[Input]:
        return self._weighted_input

    @property
    def weighted_sum(self) -> _decimal.Decimal:
        """
        A simple weighted sum calculation. Can be optimized using Vector math (Dot Product).
        """

        return sum(
            (input.computed_value for input in self._weighted_input),
            start=_decimal.Decimal(),
        )

    @property
    def output(self) -> _decimal.Decimal:
        if self._linear_output is None:
            self._linear_output = self.weighted_sum + self.bias

        assert self._linear_output is not None
        # To introduce non-linearity to the output, use one of the common activation functions.
        # An activation function is applied to the calculated linear output. Commonly used functions:
        # - Sigmoid
        # - ReLU (Rectified Linear Unit)
        # - LazyRelU (Lazy Rectified Linear Unit)
        # - Tanh (Hyperbolic Tangent)
        # - Softmax
        return _relu(self._linear_output)

    @property
    def loss(self) -> _decimal.Decimal:
        """
        Cross-entropy loss function.
        """

        if self._loss is None:
            self._loss = -self._target * _decimal.Decimal(_math.log(self.output))

        return self._loss

    @property
    def error(self) -> _decimal.Decimal:
        assert self._error is not None, "Call calculate_error to compute error first"

        return self._error

    def compute_error(self, next_layer: list["Neuron"] | None = None) -> None:
        if self._is_in_output_layer:
            self._error = _relu_dx(self.output) * (self._target - self.output)
        else:
            assert (
                next_layer is not None
            ), "Invalid next_layer for the hidden layer Neuron error calculation"

            self._error = _relu_dx(self.output) * sum(
                neuron.error * weight
                for neuron, weight in zip(next_layer, self.weight_vector)
            )


class NeuralNetwork:
    def __init__(
        self,
        *,
        bias: _decimal.Decimal,
        width: int = 3,
        depth: int = 4,
        target: list[_decimal.Decimal],
    ):
        self._bias = bias
        self._depth = depth
        self._width = width

        self._loss: _decimal.Decimal | None = None
        self._output: list[_decimal.Decimal] = []
        self._output_layer: list[Neuron] = []

        self._target = target
        self._learning_rate = _decimal.Decimal("0.1")

        self._nn_state: list[list[Neuron]] = []

        # Initialize each Neuron in the network with a bias, random inputs, and weights.
        self._nn_state = [
            [
                Neuron(bias=self._bias, target=self._target[width_idx])
                for _ in range(depth)
            ]
            for width_idx in range(width)
        ]

    def _get_layer(self, idx: int) -> list[Neuron]:
        return [self._nn_state[width_idx][idx] for width_idx in range(self._width)]

    def _compute_layer(self, layer: list[Neuron]) -> list[_decimal.Decimal]:
        return [neuron.output for neuron in layer]

    @property
    def loss(self) -> _decimal.Decimal:
        """
        Loss function for the neural network.
        Computes Squared Loss of the network's output.
        This loss function is only valid for running the NN on a single input (or input vector).
        If NN is running on multiple inputs (or input vectors) then MSE can/should be used as a loss function (?).

        Update: Using Cross-entropy loss function after converting to multi-class Neural Network.
        """

        if self._loss is None:
            self._loss = sum(
                (neuron.loss for neuron in self._output_layer),
                start=_decimal.Decimal(0),
            )

        return self._loss

    @property
    def output(self) -> list[_decimal.Decimal]:
        # TODO: The output layer typically uses a different activation function than the hidden layers (why?).
        # Update the interface to make that possible.
        if not self._output:
            self._output = self._compute_layer(self._output_layer)

        return self._output

    def _get_weights(self, *, row: int, col: int) -> list[_decimal.Decimal]:
        """
        Get the weights of the neurons in the network.
        """

        neuron = self._nn_state[row][col]
        return [input.weight for input in neuron.weighted_input]

    def _do_back_propagation(self):
        """
        Back propagation computes error for every Neuron in every layer of the Network.
        """

        for depth_idx in range(self._depth - 1, -1, -1):
            for width_idx in range(self._width):
                neuron = self._nn_state[depth_idx][width_idx]

                if depth_idx == self._depth - 1:
                    # Output layer Neuron
                    neuron.compute_error()
                else:
                    next_layer = self._get_layer(depth_idx + 1)
                    neuron.compute_error(next_layer=next_layer)

    def _adjust_weights(self):
        """
        Updates weights & biases of each Neuron in the Network using Gradient descent.
        """

        assert self.loss is not None

        self._do_back_propagation()
        for layer in self._nn_state:
            for neuron in layer:
                for input in neuron.weighted_input:
                    input.weight -= self._learning_rate * neuron.error * input.value

                neuron.bias -= self._learning_rate * neuron.error

    def _compute_layers(self, input: list[_decimal.Decimal]) -> None:
        """
        Runs a single iteration of the neural network on the input vector. Here's how the Neural Network can be visualized:
            Width = 2 (number of Neurons per layer)
            Depth = 4 (3 hidden layers, 1 output layer)
            Neuron = *

            Hidden layer 0 -> Hidden layer 1 -> Hidden layer 2 -> Output layer
                *        ->      *           ->      *         ->      *
                *        ->      *           ->      *         ->      *

        For the first iteration of this method, all weights in the network are randomized.
        After the first iteration of this method, each Neuron in the network will have calculated inputs but still have random weights.
        """
        output = _copy.deepcopy(input)

        # Top-down (vertical) Neurons creation.
        # Take a look at the visualization above.
        # The outer loop points to a (vertical) layer.
        # The inner loop initializes and computes the individual Neurons of the (vertical) layer (can be parallelized).
        # The combined output of one (vertical) layer is fed as inputs to each of the Neurons of the next (vertical) layer.
        for depth_idx in range(self._depth):
            self._output_layer = []
            is_final_output_layer = depth_idx == self._depth - 1
            # Create Neuron at each width level in this layer
            for width_idx in range(self._width):
                bias = _decimal.Decimal(_random.random())
                target = self._target[width_idx]
                weights = self._get_weights(row=width_idx, col=depth_idx)

                self._output_layer.append(
                    Neuron(
                        bias=bias,
                        target=target,
                        inputs=output,
                        weights=weights,
                        is_in_output_layer=is_final_output_layer,
                    )
                )

                self._nn_state[width_idx][depth_idx] = self._output_layer[-1]

            # Compute the output of this layer.
            output = self._compute_layer(self._output_layer)

        assert (
            len(self._output_layer) == self._width
        ), f"The output layer length {len(self._output_layer)} is not equal to the width of the network, something went wrong"

    def run(
        self,
        *,
        inputs: list[_decimal.Decimal],
        loss_tolerance: _decimal.Decimal = _decimal.Decimal("0.01"),
    ) -> None:
        print("\nNetworkGoal: Minimize loss")

        self._compute_layers(inputs)

        while self.loss > loss_tolerance:
            print(f"Loss {self.loss} > {loss_tolerance}. Output: {self.output}")

            self._adjust_weights()
            self._compute_layers(self.output)

        print(
            f"\nRun complete. Loss {self.loss} <= {loss_tolerance}. Output: {self.output}"
        )
