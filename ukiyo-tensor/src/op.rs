use std::fmt::{self, Display};

use crate::{tensor::Tensor, tensor_storage::TensorItem};

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub(crate) enum Op<T: TensorItem> {
    Add { lhs: Tensor<T>, rhs: Tensor<T> },
    Neg { input: Tensor<T> },
    Sub { lhs: Tensor<T>, rhs: Tensor<T> },
    Mul { lhs: Tensor<T>, rhs: Tensor<T> },
    Div { lhs: Tensor<T>, rhs: Tensor<T> },
    Tanh { input: Tensor<T> },
    Relu { input: Tensor<T> },
    Exp { exp: Tensor<T> },
    Powf { base: Tensor<T>, exp: Tensor<f32> },
}

impl<T: TensorItem> Op<T> {
    pub(crate) fn inputs(&self) -> impl Iterator<Item = &Tensor<T>> {
        match self {
            Op::Add { lhs, rhs }
            | Op::Sub { lhs, rhs }
            | Op::Mul { lhs, rhs }
            | Op::Div { lhs, rhs } => vec![lhs, rhs].into_iter(),
            Op::Tanh { input } | Op::Relu { input } | Op::Neg { input } => vec![input].into_iter(),
            Op::Exp { exp } => vec![exp].into_iter(),
            // TODO: any problems if exp is removed here?
            Op::Powf { base, exp: _ } => vec![base].into_iter(),
        }
    }
}

impl<T: TensorItem> Display for Op<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Add { .. } => write!(f, "+"),
            Op::Neg { .. } => write!(f, "-"),
            Op::Sub { .. } => write!(f, "-"),
            Op::Mul { .. } => write!(f, "*"),
            Op::Div { .. } => write!(f, "/"),
            Op::Tanh { .. } => write!(f, "tanh"),
            Op::Relu { .. } => write!(f, "relu"),
            Op::Exp { .. } => write!(f, "exp"),
            Op::Powf { .. } => write!(f, "^"),
        }
    }
}

pub trait Tanh {
    fn tanh(self) -> Self;
}

pub trait Relu {
    fn relu(self) -> Self;
}

pub trait Exp {
    type Output;

    fn exp(self) -> Self::Output;
}

pub trait Powf {
    type Output;

    fn powf(self, exp: f32) -> Self::Output;
}

pub trait Powi {
    type Output;

    fn powi(self, exp: i32) -> Self::Output;
}
