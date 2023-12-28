import math as _math
import copy as _copy
import decimal as _decimal
import random as _random


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
    ):
        self._bias = bias
        self._target = target

        self._loss: _decimal.Decimal | None = None
        self._linear_output: _decimal.Decimal | None = None

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
            self._linear_output = self.weighted_sum + self._bias

        return self._activation_function()

    @property
    def loss(self) -> _decimal.Decimal:
        if self._loss is None:
            self._loss = -self._target * _decimal.Decimal(_math.log(self.output))

        return self._loss

    def _activation_function(self) -> _decimal.Decimal:
        """
        To introduce non-linearity to the output, use one of the common activation functions.
        An activation function is applied to the calculated linear output. Commonly used functions:
        - Sigmoid Function
        - ReLU (Rectified Linear Unit) Function
        - LazyRelU (Lazy ReLU) Function
        - Tanh (Hyperbolic Tangent) Function
        - Softmax Function
        """

        assert (
            self._linear_output
        ), "Linear output must be calculated to pass through the activation function"

        # ReLU (Rectified Linear Unit)
        return max(_decimal.Decimal(0), self._linear_output)

    def reset_computed_values(self) -> None:
        self._loss = None
        self._linear_output = None


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

        self._loss = None
        self._output_layer: list[Neuron] = []

        self._target = target
        self._learning_rate = _decimal.Decimal("0.1")

        self._nn_state: list[list[Neuron]] = []

        # Initialize each Neuron in the network with a bias, random inputs, and weights.
        for width_idx in range(self._width):
            for depth_idx in range(self._depth):
                self._nn_state[width_idx][depth_idx] = Neuron(
                    bias=self._bias, target=self._target[width_idx]
                )

    def _get_layer(self, num: int) -> list[Neuron]:
        return [self._nn_state[width_idx][num - 1] for width_idx in range(self._width)]

    def _compute_layer(self, layer: list[Neuron]) -> list[_decimal.Decimal]:
        return [neuron.output for neuron in layer]

    @property
    def loss(self) -> _decimal.Decimal:
        if self._loss is None:
            raise ValueError("Loss not computed. Run the neural network first")

        return self._loss

    @property
    def output(self) -> list[_decimal.Decimal]:
        # TODO: The output layer typically uses a different activation function than the hidden layers (why?).
        # Update the interface to make that possible.
        return self._compute_layer(self._output_layer)

    def _relu_derivative(self, value: _decimal.Decimal) -> _decimal.Decimal:
        return (
            _decimal.Decimal(1) if value > _decimal.Decimal(0) else _decimal.Decimal(0)
        )

    def _compute_loss(self):
        """
        Loss function for the neural network.
        Computes Squared Loss of the network's output.
        This loss function is only valid for running the NN on a single input (or input vector).
        If NN is running on multiple inputs (or input vectors) then MSE can/should be used as a loss function (?).

        Update: Using Cross-entropy loss after converting to multi-class Neural Network.
        """

        if self._loss is None:
            # Cross-entropy for one class (?)
            self._loss = sum(
                (neuron.loss for neuron in self._output_layer),
                start=_decimal.Decimal(0),
            )

    def _get_weights(self, *, row: int, col: int) -> list[_decimal.Decimal]:
        """
        Get the weights of the neurons in the network.
        """

        neuron = self._nn_state[row][col]
        return [input.weight for input in neuron.weighted_input]

    def _back_propagate(self):
        """
        Back propagation to adjust weights.
        I do not understand how back propagation works yet.
        """

        # Step 1: Calculate the errors for the output layer.
        output_errors = [
            neuron.output - target_val
            for neuron, target_val in zip(self._output_layer, self._target)
        ]

        # Step 2: Calculate gradients for the weights between the last hidden layer and the output layer.
        last_hidden_layer = self._get_layer(self._depth - 2)
        for neuron in last_hidden_layer:
            for j, output_neuron in enumerate(self._output_layer):
                # Calculate the gradient for the weight connecting neuron of the last hidden layer
                # to neuron j of the output layer
                gradient = output_errors[j] * self._relu_derivative(
                    output_neuron.weighted_sum
                )
                # Adjust the weight by subtracting the product of the gradient and the learning rate
                neuron.weighted_input[j].weight -= self._learning_rate * gradient

        # TODO: Step 3: Propagate errors back through the hidden layers.

    def _grandient_descent(self):
        """
        Gradient Descent is used to adjust the weights of the neurons in the network.
        It is used to minimize the loss value.
        """

        # TODO: Write the code for Gradient Descent here.

    def _adjust_weights(self):
        assert self.loss

        # TODO: Use Back Propagation to adjust the weight according to the loss value.
        # TODO: Define Learning Rate to control the degree of weight adjustments.

        for width_idx in range(self._width):
            for depth_idx in range(self._depth):
                neuron = self._nn_state[width_idx][depth_idx]
                prev_weights = neuron.weight_vector
                # What's next?

    def _compute_layers(self, input: list[_decimal.Decimal]) -> None:
        """
        Runs a single iteration of the neural network on the input vector. Here's how the Neural Network can be visualized:
            Width = 3 (number of Neurons per layer)
            Depth = 4 (1 input layer, 2 hidden layers, 1 output layer)
            Neuron = *

            Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
                *       ->      *         ->      *         ->      *
                *       ->      *         ->      *         ->      *
                *       ->      *         ->      *         ->      *

        For the first iteration of the neural network, all weights in the network are randomized.
        """
        output = _copy.deepcopy(input)

        # Using a top-down approach for computing the output.
        # Take a look at the visualization above.
        # The outer loop points to a (vertical) layer.
        # The inner loop initializes and computes the individual Neurons of the (vertical) layer (can be parallelized).
        # The combined output of one (vertical) layer is fed as inputs to each of the Neurons of the next (vertical) layer.
        # After a single run of this method, each Neuron in the network will have inputs and weights
        for depth_idx in range(self._depth):
            # Input layer initialization.
            self._output_layer = []
            for width_idx in range(self._width):
                # Input layer weights initialization.
                weights = self._get_weights(row=width_idx, col=depth_idx)

                self._output_layer.append(
                    Neuron(
                        bias=self._bias,
                        target=self._target[width_idx],
                        inputs=output,
                        weights=weights,
                    )
                )

                self._nn_state[width_idx][depth_idx] = self._output_layer[-1]

            # Compute the output of this layer.
            output = self._compute_layer(self._output_layer)

        assert (
            len(self._output_layer) == self._width
        ), f"The output layer length {len(self._output_layer)} is not equal to the width of the network, something went wrong"

    def run(self, *, inputs: list[_decimal.Decimal]) -> None:
        loss_tolerance = _decimal.Decimal(0.01)

        print("\nCompute Goal: Minimize loss")

        self._compute_layers(inputs)
        self._compute_loss()

        while self.loss > loss_tolerance:
            print(f"Loss {self.loss} > {loss_tolerance}")

            self._adjust_weights()
            self._compute_layers(self.output)
            self._compute_loss()

        print(f"Run complete. Loss {self.loss} <= {loss_tolerance}")
