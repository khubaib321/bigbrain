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
        inputs: list[_decimal.Decimal] = [],
        weights: list[_decimal.Decimal] = [],
    ):
        self._bias = bias
        self._linear_output = None

        if not weights:
            self._input_vector = [
                Input(value=v, weight=_decimal.Decimal(_random.random()))
                for v in inputs
            ]
        else:
            if len(inputs) != len(weights):
                raise ValueError(
                    "Number of inputs and weights must be equal. "
                    f"Inputs: {len(inputs)}, Weights: {len(weights)}"
                )

            self._input_vector = [
                Input(value=v, weight=w) for v, w in zip(inputs, weights)
            ]

    @property
    def input_vector(self) -> list[Input]:
        return self._input_vector

    @property
    def _weighted_sum(self) -> _decimal.Decimal:
        """
        A simple weighted sum calculation. Can be optimized using Vector math (Dot Product).
        """
        
        return sum(
            (input.computed_value for input in self._input_vector),
            start=_decimal.Decimal(),
        )

    def _activation_function(self) -> _decimal.Decimal:
        """
        To introduce non-linearity to the output, use one of the common activation functions.
        An activation function is applied to the calculated linear output. Commonly used functions:
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
        return max(_decimal.Decimal(0), self._linear_output)

    @property
    def output(self) -> _decimal.Decimal:
        if self._linear_output is None:
            self._linear_output = self._weighted_sum + self._bias

        return self._activation_function()


class NeuralNetwork:
    def __init__(
        self,
        *,
        bias: _decimal.Decimal,
        width: int = 3,
        depth: int = 3,
        learning_rate: _decimal.Decimal = _decimal.Decimal("0.05"),
    ):
        self._bias = bias
        self._depth = depth
        self._width = width

        self._loss = None
        self._output = None
        self._learning_rate = learning_rate

        self._nn_state: list[list[Neuron]] = []

        # Initialize each Neuron in the network with a bias, random inputs, and weights.
        for _ in range(self._depth):
            self._nn_state.append([Neuron(bias=self._bias) for _ in range(self._width)])

    def _compute_layer(self, layer: list["Neuron"]) -> list[_decimal.Decimal]:
        return [neuron.output for neuron in layer]

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
        If NN is running on multiple inputs (or input vectors) then MSE can/should be used as a loss function.
        """

        # TODO: I have yet to learn what it means to run a single input vs multiple inputs.
        self._loss = (self.output - target_ouput) ** 2

    def _backpropagation(self, target_output: _decimal.Decimal):
        """
        Back Propagation is used to adjust the weights of the neurons in the network.
        It is used to minimize the loss value.
        """

        # TODO: Write the code for Back Propagation here.

    def _grandient_descent(self):
        """
        Gradient Descent is used to adjust the weights of the neurons in the network.
        It is used to minimize the loss value.
        """

        # TODO: Write the code for Gradient Descent here.

    def _get_weights(self, *, row: int, col: int) -> list[_decimal.Decimal]:
        """
        Get the weights of the neurons in the network.
        """

        neuron = self._nn_state[row][col]

        current_weights = [input.weight for input in neuron.input_vector]

        if self._loss and self._output:
            # TODO: Adjust weights (?).
            # TODO: Use Back Propagation to adjust the weight according to the loss value.
            # TODO: Define Learning Rate to control the degree of weight adjustments.
            pass

        return current_weights

    def compute_output(self, inputs: list[_decimal.Decimal]) -> None:
        """
        Runs a single iteration of the neural network on the input vector. Here's how the Neural Network can be visualized:
            Width = 3
            Depth = 2
            Neuron = *

            Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
                *       ->      *         ->      *         ->      *       \
                *       ->      *         ->      *         ->      *       ->      * (output neuron)
                *       ->      *         ->      *         ->      *       /
        """

        # TODO: The output layer typically uses a different activation function than the hidden layers (why?).
        # Adjust the Neuron interface to allow the use of different activation functions.

        output = _copy.deepcopy(inputs)

        # Using a top-down approach for computing the output.
        # Take a look at the visualization above.
        # The outer loop points to a (vertical) layer.
        # The inner loop initializes and computes the individual Neurons of the (vertical) layer (can be parallelized).
        # The combined output of one (vertical) layer is fed as inputs to each of the Neurons of the next (vertical) layer.
        # After a single run of this method, each Neuron in the network will have inputs and weights
        for layer_idx in range(self._depth):
            # Input layer initialization.
            input_layer = []
            for level_idx in range(self._width):
                # Input layer weights initialization.
                weights = self._get_weights(row=level_idx, col=layer_idx)

                input_layer.append(
                    Neuron(bias=self._bias, inputs=output, weights=weights)
                )

                self._nn_state[level_idx][layer_idx] = input_layer[-1]

            # Compute the output of the layer. Feed the output of the current layer to the next layer.
            output = self._compute_layer(input_layer)

        weights = self._get_weights(row=self._width - 1, col=self._depth - 1)
        self._output = Neuron(bias=self._bias, inputs=output, weights=weights).output

    def run(
        self, *, inputs: list[_decimal.Decimal], target_output: _decimal.Decimal
    ) -> None:
        min_loss = _decimal.Decimal(0.01) * target_output

        print("\nCompute Goal: Minimize loss")

        self.compute_output(inputs)
        self._compute_loss(target_output)

        # TODO: Verify: Loop will not run indefinitely, right?
        while self.loss > min_loss:
            print(f"Current Loss {self.loss} > Min Loss {min_loss}")

            self.compute_output(inputs)
            self._compute_loss(target_output)

        print(f"Current Loss {self.loss} <= Min Loss {min_loss}")
