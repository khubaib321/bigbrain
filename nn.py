import math as _math
import copy as _copy
import random as _random
import decimal as _decimal

_DEFAULT_BIAS = _decimal.Decimal(0.1)


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


def _softmax(logits: list[_decimal.Decimal]) -> list[_decimal.Decimal]:
    # Set the precision for Decimal calculations
    _decimal.getcontext().prec = 28

    # Convert logits to Decimal if they are not already
    logits = [_decimal.Decimal(logit) for logit in logits]

    # Compute the maximum logit value for numerical stability
    max_logit = max(logits)

    # Compute the exponential of each logit, adjusted by the max logit value
    exps = [_decimal.getcontext().exp(logit - max_logit) for logit in logits]

    # Sum of all exponentiated logits
    sum_of_exps = sum(exps)

    # Normalize each exponentiated logit
    softmax_output = [exp / sum_of_exps for exp in exps]

    return softmax_output


def _softmax_dx(value: _decimal.Decimal) -> _decimal.Decimal:
    return value


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
            assert len(inputs) == len(
                weights
            ), f"Number of inputs and weights must be equal. Inputs: {len(inputs)}, Weights: {len(weights)}"

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
        self._linear_output = self.weighted_sum + self.bias

        assert self._linear_output is not None
        # To introduce non-linearity to the output, use one of the common activation functions.
        # An activation function is applied to the calculated linear output. Commonly used functions:
        # - Sigmoid
        # - ReLU (Rectified Linear Unit)
        # - LazyRelU (Lazy Rectified Linear Unit)
        # - Tanh (Hyperbolic Tangent)
        # - Softmax

        if not self.is_output_neuron:
            # Use ReLU for hidden layers
            return _relu(self._linear_output)

        return self._linear_output

    @property
    def error(self) -> _decimal.Decimal:
        assert self._error is not None, "Call calculate_error to compute error first"

        return self._error

    @property
    def is_output_neuron(self) -> bool:
        return self._is_in_output_layer

    def compute_error(self, next_layer: list["Neuron"] | None = None) -> None:
        if self.is_output_neuron:
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
        depth: int = 4,
        target: list[_decimal.Decimal],
    ):
        self._depth = depth
        self._width = len(target)

        self._loss: _decimal.Decimal | None = None
        self._output: list[_decimal.Decimal] = []
        self._output_layer: list[Neuron] = []

        self._target = target
        self._learning_rate = _decimal.Decimal(0.1)

        self._nn_state: list[list[Neuron]] = []

        # Initialize each Neuron in the network with a bias, random inputs, and weights.
        self._nn_state = [
            [
                Neuron(bias=_DEFAULT_BIAS, target=self._target[width_idx])
                for _ in range(self._depth)
            ]
            for width_idx in range(self._width)
        ]

    def _get_layer(self, idx: int) -> list[Neuron]:
        return [self._nn_state[width_idx][idx] for width_idx in range(self._width)]

    def _compute_layer(self, layer: list[Neuron]) -> list[_decimal.Decimal]:
        return [neuron.output for neuron in layer]
    
    def _one_hot_encode(self, target: _decimal.Decimal):
        """
        Convert a class label into one-hot encoded format.
        """

        # TODO: One-hot encoding not correct.
        one_hot = [_decimal.Decimal(0) for _ in range(len(self._target))]
        one_hot[index] = self._target[index]
        return one_hot
    
    def _compute_loss(self, targets: list[_decimal.Decimal], outputs: list[_decimal.Decimal]):
        """
        Compute the cross-entropy loss for multi-class classification.
        """
        loss = _decimal.Decimal(0)
        for target, output in zip(targets, outputs):
            target_one_hot = self._one_hot_encode(target)
            # TODO: One-hot encoding not correct.
            for t, o in zip(target_one_hot, output):
                loss -= t * _decimal.Decimal(_math.log(o + _decimal.Decimal(1e-10)))
        return loss

    @property
    def loss(self) -> _decimal.Decimal:
        """
        Loss function for the neural network.
        Computes Squared Loss of the network's output.
        This loss function is only valid for running the NN on a single input (or input vector).
        If NN is running on multiple inputs (or input vectors) then MSE can/should be used as a loss function (?).

        Update: Using Cross-entropy loss function after converting to multi-class Neural Network.
        """

        self._loss = self._compute_loss(self._target, self.output)
        return self._loss

    @property
    def output(self) -> list[_decimal.Decimal]:
        # TODO: The output layer typically uses a different activation function than the hidden layers (why?).
        # Update the interface to make that possible.
        outputs = self._compute_layer(self._output_layer)
        self._output = _softmax(outputs)

        return self._output

    def _get_bias(self, *, row: int, col: int) -> _decimal.Decimal:
        return self._nn_state[row][col].bias

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
            for neuron in self._get_layer(depth_idx):
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
                    input.weight += self._learning_rate * neuron.error * input.value

                neuron.bias += self._learning_rate * neuron.error

    def print_state(self) -> None:
        print(f"Neural Network state: Width {self._width}, Depth {self._depth}")
        for level in self._nn_state:
            print(
                [
                    f"{neuron.output} ({'o'* neuron.is_output_neuron or 'h'})"
                    for neuron in level
                ]
            )

    def _compute_layers(self, input: list[_decimal.Decimal], iteration: int) -> None:
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
        self._output_layer = []
        output = _copy.deepcopy(input)
        print(f"Iteration: {iteration}")

        # Top-down (vertical) Neurons creation.
        # Take a look at the visualization above.
        # The outer loop points to a (vertical) layer.
        # The inner loop initializes and computes the individual Neurons of the (vertical) layer (can be parallelized).
        # The combined output of one (vertical) layer is fed as inputs to each of the Neurons of the next (vertical) layer.
        for depth_idx in range(self._depth):
            layer = []
            is_final_output_layer = depth_idx == self._depth - 1
            # Create Neuron at each width level in this layer
            for width_idx in range(self._width):
                target = self._target[width_idx]
                bias = self._get_bias(row=width_idx, col=depth_idx)
                weights = self._get_weights(row=width_idx, col=depth_idx)
                print(f"Weights: {weights}")

                neuron = Neuron(
                    bias=bias,
                    target=target,
                    inputs=output,
                    weights=weights,
                    is_in_output_layer=is_final_output_layer,
                )

                layer.append(neuron)
                if is_final_output_layer:
                    self._output_layer.append(neuron)

                self._nn_state[width_idx][depth_idx] = layer[-1]

            # Compute the output of this layer
            output = self._compute_layer(layer)

        assert (
            len(self._output_layer) == self._width
        ), f"The output layer length {len(self._output_layer)} is not equal to the width of the network, something went wrong"

    def _compute_inputs(self, inputs: list[_decimal.Decimal]) -> list[_decimal.Decimal]:
        layer = [
            Neuron(bias=_DEFAULT_BIAS, inputs=inputs, target=target)
            for target in self._target
        ]

        return self._compute_layer(layer)

    def run(
        self,
        *,
        inputs: list[_decimal.Decimal],
        loss_tolerance: _decimal.Decimal = _decimal.Decimal(0.01),
    ) -> None:
        print("Network Goal: Minimize loss")

        iteration = 1
        inputs_computed = self._compute_inputs(inputs)
        self._compute_layers(inputs_computed, iteration)
        current_output = [f"{o}" for o in self.output]

        while abs(self.loss) > loss_tolerance:
            breakpoint()
            print(f"Loss {self.loss} > {loss_tolerance}. Output: {current_output}")

            iteration += 1
            self.print_state()
            self._adjust_weights()
            self._compute_layers(self.output, iteration)
            current_output = [f"{o}" for o in self.output]

        print(
            f"\nRun complete. Loss {self.loss} <= {loss_tolerance}. Output: {current_output}"
        )
