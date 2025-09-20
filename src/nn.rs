use rand_distr::{Distribution, Uniform};

use crate::engine::{Powf, Tanh, Value};

pub trait Parameters {
    fn parameters(&self) -> Vec<&Value>;
    /// Sets all gradients of the parameters to zero
    fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.0));
    }
}

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    /// Creates a new [`Neuron`] with `nin` inputs.
    fn new(nin: usize) -> Self {
        let mut rng = rand::rng();
        let die: Uniform<f32> = Uniform::new_inclusive(-1.0, 1.0)
            .expect("Failed to create uniform distribution: invalid range");

        Self {
            weights: (0..nin).map(|_| Value::new(die.sample(&mut rng))).collect(),
            bias: Value::new(die.sample(&mut rng)),
        }
    }

    /// Returns the dot product of inputs and weights plus bias
    /// inputs.len must == self.weights.length
    fn forward(&self, inputs: &[f32]) -> Value {
        let raw = self
            .weights
            .iter()
            .cloned()
            .zip(inputs.iter().cloned())
            .map(|(wi, xi)| wi * xi)
            .sum::<Value>()
            + self.bias.clone();
        // pass through activation function
        raw.tanh()
    }
}

impl Parameters for Neuron {
    /// Returns the tuneable knobs of the neuron: weights and bias
    fn parameters(&self) -> Vec<&Value> {
        let mut params = self.weights.iter().collect::<Vec<_>>();
        params.push(&self.bias);
        params
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a new [`Layer`] of `nout` neurons, each with `nin` inputs (dimensionality)
    fn new(nin: usize, nout: usize) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    fn forward(&self, inputs: &[f32]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }
}

impl Parameters for Layer {
    /// Returns the parameters of all neurons
    fn parameters(&self) -> Vec<&Value> {
        self.neurons
            .iter()
            .flat_map(|n| n.parameters().into_iter())
            .collect()
    }
}

pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Creates a new [`MLP`] with `nin` neurons in the input layer and `nouts` as a list of number of
    /// neurons in each subsequent layer
    ///
    /// Invariant: number of neurons in layer n == input dim of layer n+1
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let sz = std::iter::once(nin)
            .chain(nouts.iter().cloned())
            .collect::<Vec<_>>();
        Self {
            layers: (0..nouts.len())
                .map(|i| Layer::new(sz[i], sz[i + 1]))
                .collect(),
        }
    }

    pub fn forward(&self, inputs: &[f32]) -> Vec<Value> {
        let mut x = inputs.iter().cloned().map(Value::new).collect::<Vec<_>>();
        for layer in &self.layers {
            x = layer.forward(x.iter().map(|v| v.data()).collect::<Vec<_>>().as_slice());
        }
        x
    }

    pub fn train(&self, xs: &[&[f32]], ys: &[f32]) {
        // forward pass
        let ypred = xs.iter().map(|x| self.forward(x)).collect::<Vec<_>>();
        let loss = ys
            .iter()
            .zip(ypred)
            .map(|(ygt, mut yout)| (yout.pop().expect("yout cannot be empty") - *ygt).powf(2.0))
            .sum::<Value>();
        println!("loss = {:.4}", loss.data());

        // flush the gradients so gradients from previous step do not accumulate
        self.zero_grad();

        // backward pass
        loss.backward();

        // update params
        self.parameters().iter().for_each(|p| {
            p.increase_data(-0.1 * p.grad());
        });
    }
}

impl Parameters for Mlp {
    /// Returns the parameters of all layers
    fn parameters(&self) -> Vec<&Value> {
        self.layers
            .iter()
            .flat_map(|l| l.parameters().into_iter())
            .collect()
    }
}
