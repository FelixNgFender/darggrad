# ukiyo

Autograd engine made in a day.

Supported operations: sum, negate, multiply, divide, power, and ReLu (squash at
less than zero).

Works on scalar values only.

Also includes a two-layer multi-layer perceptron (MLP) neural network example.

## Notes

Character-level language model predicts the next character in a sequence given
some concrete characters before it.

_Intuition_: A word in itself already packs a lot of
examples/structure/statistics a model can pick up on.

## Bigram

One character simply predicts the next one using a lookup table of counts.
