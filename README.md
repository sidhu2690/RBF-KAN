# Guide for Using RBFKAN

The `RBFKAN` class is a PyTorch module that implements a Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN). This module combines the traditional neural network approach with a radial basis function (RBF) kernel to capture non-linear relationships in the input data. It is designed to be used as a layer in a larger neural network architecture.

## It's one of the fastest KAN and has a speed of 60its/s

## Class RBFLinear

The `RBFLinear` class is a sub-module that implements the RBF kernel transformation of the input data. It takes the following arguments:

- `in_features`: The number of input features.
- `out_features`: The number of output features.
- `grid_min` (default=-2.0): The minimum value of the grid for the RBF kernel.
- `grid_max` (default=2.0): The maximum value of the grid for the RBF kernel.
- `num_grids` (default=8): The number of grid points for the RBF kernel.
- `spline_weight_init_scale` (default=0.1): The scale factor for initializing the spline weights.

The `forward` method of this class applies the RBF kernel transformation to the input data.

## Class RBFKANLayer

The `RBFKANLayer` class is the main building block of the `RBFKAN` module. It combines the `RBFLinear` transformation with a traditional linear layer. It takes the following arguments:

- `input_dim`: The number of input features.
- `output_dim`: The number of output features.
- `grid_min`, `grid_max`, `num_grids`, and `spline_weight_init_scale`: Same as in `RBFLinear`.
- `use_base_update` (default=True): Whether to use the traditional linear layer in addition to the RBF kernel.
- `base_activation` (default=nn.SiLU()): The activation function for the traditional linear layer.

The `forward` method of this class applies the RBF kernel transformation and, optionally, the traditional linear layer to the input data.

## Class RBFKAN

The `RBFKAN` class is the main module that combines multiple `RBFKANLayer` instances into a larger neural network architecture. It takes the following arguments:

- `layers_hidden`: A list of integers representing the number of features in each hidden layer, including the input and output layers.
- `grid_min`, `grid_max`, `num_grids`, `use_base_update`, `base_activation`, and `spline_weight_init_scale`: Same as in `RBFKANLayer`.

The `forward` method of this class applies the sequence of `RBFKANLayer` instances to the input data, passing the output of one layer as the input to the next layer.

## Usage Example

```python
# Define the input and output dimensions
input_dim = 10
output_dim = 5

# Define the hidden layer dimensions
hidden_dims = [16, 32, 16]

# Create an RBFKAN instance
model = RBFKAN([input_dim] + hidden_dims + [output_dim])

# Forward pass with some input data
x = torch.randn(batch_size, input_dim)
y = model(x)
