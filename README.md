# darggrad

Autograd engine made in a day.

Supported operations: sum, negate, multiply, divide, power, and ReLu (squash at
less than zero).

Works on scalar values only.

## Notes

### Backpropagation

An algorithm that evaluates gradient of the loss function with respect to the
neural network's weights. Allows you to minimize loss function by iteratively
tune the weights, improving accuracy of network.

Originated in 1964, it doesn't involve neural network, but happens to be useful
so people use it for training NNs.

In the context of neural networks, backprop is only meaningful when training
using loss (i.e., gradient descent technically). There are many other training
methods that don't rely on loss function, and thus backprop.

#### How it works

Backprop works in 2 phases:

1. The forward pass: traces the operations performed on the tensors and
   constructs a computational DAG
1. The backward pass: traverses the DAG from the output node to the input nodes,
   recursively applying the chain rule to compute gradients.

This information is crucial as it allows us to see how much the inputs are
affecting the outputs:

$$
a = Value(1)\\
b = Value(2)\\
a + b = c\\
c.value == 3\\
c.backward()\\
a.grad = \frac{dc}{da} = ...\\
b.grad = \frac{db}{da} = ...\\
$$

#### The chain rule

[Wikipedia](https://en.wikipedia.org/wiki/Chain_rule)

#### Tensors

Biggy arrays of scalars, packed together for efficiency when training on GPUs.
Not useful for demonstrating backprop, so we only support scalars in darggrad.
