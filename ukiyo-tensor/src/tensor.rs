use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    rc::Rc,
};

use crate::{Exp, Powf, Powi, Relu, Tanh, TensorImpl, op::Op, tensor_storage::TensorItem};

#[derive(PartialEq, Eq, Hash, Debug)]
struct TensorInner<T: TensorItem> {
    op: Option<Op<T>>,
    data: TensorImpl<T>,
    /// Gradient relative to the first caller of `backward()`
    grad: TensorImpl<T>,
}

#[derive(Clone, Debug)]
pub struct Tensor<T: TensorItem> {
    /// Allows a tensor to be shared and mutated across multiple owners, essentially allowing
    /// construction of a DAG computation graph (a tensor may contribute to more than 1 output).
    inner: Rc<RefCell<TensorInner<T>>>,
}

impl<T: TensorItem> From<TensorImpl<T>> for Tensor<T> {
    fn from(data: TensorImpl<T>) -> Self {
        let shape = data.shape.shape.clone();
        Self {
            inner: Rc::new(RefCell::new(TensorInner {
                op: None,
                data,
                grad: TensorImpl::zeros(shape),
            })),
        }
    }
}

impl<T: TensorItem> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T: TensorItem> Eq for Tensor<T> {}

impl<T: TensorItem> std::hash::Hash for Tensor<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // hash the pointer address instead of the inner mutating contents
        // for autodiff, we don’t care about value-equality, only care about node identity.
        Rc::as_ptr(&self.inner).hash(state);
    }
}

impl<T: TensorItem> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(TensorInner {
                op: None,
                data: TensorImpl::new(data, shape.clone()),
                grad: TensorImpl::zeros(shape),
            })),
        }
    }

    fn with_op(self, op: Option<Op<T>>) -> Self {
        self.inner.borrow_mut().op = op;
        self
    }

    pub fn data(&self) -> TensorImpl<T> {
        self.inner.borrow().data.clone()
    }

    pub fn grad(&self) -> TensorImpl<T> {
        self.inner.borrow().grad.clone()
    }

    fn op(&self) -> Option<Op<T>> {
        self.inner.borrow().op.clone()
    }
}

impl<T> Tensor<T>
where
    T: TensorItem
        + AddAssign
        + Sub<Output = T>
        + Neg<Output = T>
        + Div<Output = T>
        + Powi<Output = T>
        + Powf<Output = T>,
{
    pub fn increase_data(&self, val: TensorImpl<T>) {
        self.inner.borrow_mut().data += val;
    }

    pub fn set_grad(&self, val: TensorImpl<T>) {
        self.inner.borrow_mut().grad = val;
    }

    fn increase_grad(&self, val: TensorImpl<T>) {
        self.inner.borrow_mut().grad += val;
    }

    fn _backward(&self) {
        // assume self.grad has been initialized
        if let Some(op) = &self.op() {
            match op {
                Op::Add { lhs, rhs } => {
                    // 1.0 is local gradient, times with self.grad() for chain rule
                    lhs.increase_grad(self.grad());
                    rhs.increase_grad(self.grad());
                }
                Op::Neg { input } => {
                    input.increase_grad(-self.grad());
                }
                Op::Sub { lhs, rhs } => {
                    lhs.increase_grad(self.grad());
                    rhs.increase_grad(-self.grad());
                }
                Op::Mul { lhs, rhs } => {
                    lhs.increase_grad(rhs.data() * self.grad());
                    rhs.increase_grad(lhs.data() * self.grad());
                }
                Op::Div { lhs, rhs } => {
                    // c = a/b
                    // dc/da = 1/b
                    // dc/db = -a/(b^2)
                    lhs.increase_grad(self.grad() / rhs.data());
                    rhs.increase_grad((-lhs.data() / rhs.data().powi(2)) * self.grad());
                }
                Op::Tanh { input } => {
                    // y = tanhx
                    // dy/dx = 1 - y^2
                    input.increase_grad(
                        (TensorImpl::new(vec![1.0f32], vec![1]) - self.data().powi(2))
                            * self.grad(),
                    );
                }
                Op::Relu { input } => {
                    // y = relu(x) = max(0, x)
                    // dy/dx = 1 if x > 0, 0 otherwise
                    input.increase_grad((if input.data() > 0.0 { 1.0 } else { 0.0 }) * self.grad());
                }
                Op::Exp { exp } => {
                    // y = expx
                    // dy/dx = y
                    exp.increase_grad(self.data() * self.grad());
                }
                Op::Powf { base, exp } => {
                    // y = base^exp
                    // dy/dbase = exp * base^(exp-1)
                    base.increase_grad(
                        exp.data() * base.data().powf(exp.data() - 1.0) * self.grad(),
                    );
                }
            }
        }
    }

    fn topo(&self) -> impl DoubleEndedIterator<Item = Tensor<T>> {
        #[allow(
            clippy::mutable_key_type,
            reason = "Using identity of Value for visited set"
        )]
        fn build_topo<T: TensorItem>(
            node: &Tensor<T>,
            visited: &mut HashSet<Tensor<T>>,
            topo: &mut Vec<Tensor<T>>,
        ) {
            if visited.insert(node.clone()) {
                if let Some(op) = (*node).op() {
                    for input in op.inputs() {
                        build_topo(input, visited, topo);
                    }
                }
                topo.push(node.clone());
            }
        }

        let mut topo = Vec::new();
        #[allow(
            clippy::mutable_key_type,
            reason = "Using identity of Tensor for visited set"
        )]
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);
        topo.into_iter()
    }

    pub fn backward(&self) {
        // mark the top of the computation graph
        self.set_grad(TensorImpl::ones(self.data().shape.shape));
        // topo sort to ensure that when we compute gradient for a node, all gradients of its children have
        // already been computed
        self.topo().rev().for_each(|node| {
            node._backward();
        });
    }
}

impl<T: TensorItem> Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_with_indent<T: TensorItem>(
            value: &Tensor<T>,
            f: &mut fmt::Formatter<'_>,
            indent: usize,
        ) -> fmt::Result {
            for _ in 0..indent {
                write!(f, "    ")?; // four spaces per indent level
            }

            match &value.op() {
                Some(op) => writeln!(
                    f,
                    "Value(data={}, grad={}, op={})",
                    value.data(),
                    value.grad(),
                    op
                )?,
                None => writeln!(f, "Value(data={}, grad={})", value.data(), value.grad())?,
            }

            if let Some(op) = &value.op() {
                match op {
                    Op::Add { lhs, rhs }
                    | Op::Sub { lhs, rhs }
                    | Op::Mul { lhs, rhs }
                    | Op::Div { lhs, rhs }
                        if lhs == rhs =>
                    {
                        fmt_with_indent(lhs, f, indent + 1)?;
                    }
                    Op::Add { lhs, rhs }
                    | Op::Sub { lhs, rhs }
                    | Op::Mul { lhs, rhs }
                    | Op::Div { lhs, rhs } => {
                        fmt_with_indent(lhs, f, indent + 1)?;
                        fmt_with_indent(rhs, f, indent + 1)?;
                    }
                    Op::Neg { input } | Op::Tanh { input } | Op::Relu { input } => {
                        fmt_with_indent(input, f, indent + 1)?;
                    }
                    Op::Exp { exp } => {
                        fmt_with_indent(exp, f, indent + 1)?;
                    }
                    Op::Powf { base, exp } => {
                        fmt_with_indent(base, f, indent + 1)?;
                        fmt_with_indent(exp, f, indent + 1)?;
                    }
                }
            }

            Ok(())
        }

        fmt_with_indent(self, f, 0)
    }
}

impl<T: TensorItem + Add> Add for Tensor<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Into::<Self>::into(self.data() + rhs.data()).with_op(Op::Add { lhs: self, rhs }.into())
    }
}

impl<T: TensorItem + Neg<Output = T>> Neg for Tensor<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor::from(-self.data()).with_op(Op::Neg { input: self }.into())
    }
}

impl<T: TensorItem + Sub<Output = T>> Sub for Tensor<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::from(self.data() - rhs.data()).with_op(Op::Sub { lhs: self, rhs }.into())
    }
}

impl<T: TensorItem + Mul<Output = T>> Mul for Tensor<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::from(self.data() * rhs.data()).with_op(Op::Mul { lhs: self, rhs }.into())
    }
}

impl<T: TensorItem + Div<Output = T>> Div for Tensor<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::from(self.data() / rhs.data()).with_op(Op::Div { lhs: self, rhs }.into())
    }
}

impl<T: TensorItem + Exp<Output = T>> Exp for Tensor<T> {
    type Output = Self;

    fn exp(self) -> Self::Output {
        Tensor::from(self.data().exp()).with_op(Op::Exp { exp: self }.into())
    }
}

impl<T: TensorItem + Tanh> Tanh for Tensor<T> {
    fn tanh(self) -> Self {
        Tensor::from(self.data().tanh()).with_op(Op::Tanh { input: self }.into())
    }
}

impl<T: TensorItem + Relu> Relu for Tensor<T> {
    fn relu(self) -> Self {
        Tensor::from(self.data().relu()).with_op(Op::Relu { input: self }.into())
    }
}

impl<T: TensorItem + Powf<Output = T>> Powf for Tensor<T> {
    type Output = Self;

    fn powf(self, exp: f32) -> Self::Output {
        Tensor::from(self.data().powf(exp)).with_op(
            Op::Powf {
                base: self,
                exp: Tensor::new(vec![exp], vec![1]),
            }
            .into(),
        )
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_backward_single_add() {
//         let a = Tensor::new(2.0);
//         let b = Tensor::new(3.0);
//         let c = a.clone() + b.clone();
//         c.backward();
//
//         assert_eq!(c.grad(), 1.0);
//         assert_eq!(a.grad(), 1.0);
//         assert_eq!(b.grad(), 1.0);
//     }
//
//     #[test]
//     fn test_backward_single_neg() {
//         let a = Tensor::new(5.0);
//         let b = -a.clone();
//         b.backward();
//
//         assert_eq!(b.data(), -5.0);
//         assert_eq!(b.grad(), 1.0);
//         assert_eq!(a.grad(), -1.0);
//     }
//
//     #[test]
//     fn test_backward_single_sub() {
//         let a = Tensor::new(2.0);
//         let b = Tensor::new(3.0);
//         let c = a.clone() - b.clone();
//         c.backward();
//
//         assert_eq!(c.grad(), 1.0);
//         assert_eq!(a.grad(), 1.0);
//         assert_eq!(b.grad(), -1.0);
//     }
//
//     #[test]
//     fn test_backward_single_exp() {
//         let a = Tensor::new(2.0);
//         let b = a.clone().exp();
//         b.backward();
//
//         let tol = 1e-6;
//         assert!((b.data() - 2.0_f32.exp()).abs() < tol);
//         assert_eq!(b.grad(), 1.0);
//         assert!((a.grad() - 2.0_f32.exp()).abs() < tol); // d/dx(e^x) = e^x
//     }
//
//     #[test]
//     fn test_backward_single_square() {
//         let a = Tensor::new(3.0);
//         let b = a.clone().powf(2.0);
//         b.backward();
//
//         assert_eq!(b.data(), 9.0);
//         assert_eq!(b.grad(), 1.0);
//         assert_eq!(a.grad(), 6.0);
//     }
//
//     #[test]
//     fn test_square_negative_base() {
//         let a = Tensor::new(-2.0);
//         let b = a.clone().powf(2.0);
//         b.backward();
//
//         assert_eq!(b.data(), 4.0);
//         assert_eq!(a.grad(), -4.0);
//     }
//
//     #[test]
//     fn test_backward_single_mul() {
//         let a = Tensor::new(2.0);
//         let b = Tensor::new(3.0);
//         let c = a.clone() * b.clone();
//         c.backward();
//
//         assert_eq!(c.grad(), 1.0);
//         assert_eq!(a.grad(), 3.0);
//         assert_eq!(b.grad(), 2.0);
//     }
//
//     #[test]
//     fn test_chain_rule_computation() {
//         // ((a * b) + c) * d
//         let a = Tensor::new(1.0);
//         let b = Tensor::new(2.0);
//         let c = Tensor::new(3.0);
//         let d = Tensor::new(4.0);
//
//         let ab = a.clone() * b.clone(); // 2
//         let abc = ab.clone() + c.clone(); // 5
//         let result = abc.clone() * d.clone(); // 20
//         result.backward();
//
//         assert_eq!(result.data(), 20.0);
//         assert_eq!(result.grad(), 1.0);
//         assert_eq!(abc.grad(), 4.0);
//         assert_eq!(d.grad(), 5.0);
//         assert_eq!(ab.grad(), 4.0);
//         assert_eq!(c.grad(), 4.0);
//         assert_eq!(a.grad(), 8.0); // chain rule: 4 * b
//         assert_eq!(b.grad(), 4.0); // chain rule: 4 * a
//     }
//
//     #[test]
//     fn test_multiple_operations() {
//         // a + b * c
//         let a = Tensor::new(1.0);
//         let b = Tensor::new(2.0);
//         let c = Tensor::new(3.0);
//
//         let bc = b.clone() * c.clone(); // 6
//         let result = a.clone() + bc.clone(); // 7
//         result.backward();
//
//         assert_eq!(result.data(), 7.0);
//         assert_eq!(result.grad(), 1.0);
//         assert_eq!(a.grad(), 1.0);
//         assert_eq!(bc.grad(), 1.0);
//         assert_eq!(b.grad(), 3.0);
//         assert_eq!(c.grad(), 2.0);
//     }
//
//     #[test]
//     fn test_square_chain_rule() {
//         let a = Tensor::new(2.0);
//         let b = Tensor::new(3.0);
//         let ab = a.clone() * b.clone(); // 6
//         let result = ab.clone().powf(2.0); // 36
//         result.backward();
//
//         assert_eq!(result.data(), 36.0);
//         assert_eq!(ab.grad(), 12.0); // 2 * 6 = 12
//         assert_eq!(a.grad(), 36.0); // 12 * 3 = 36
//         assert_eq!(b.grad(), 24.0); // 12 * 2 = 24
//     }
//
//     #[test]
//     fn test_shared_node() {
//         let a = Tensor::new(3.0);
//         let b = a.clone() + a.clone();
//         b.backward();
//
//         assert_eq!(a.grad(), 2.0);
//     }
//
//     #[test]
//     fn test_shared_node_complex() {
//         let a = Tensor::new(-2.0);
//         let b = Tensor::new(3.0);
//         let d = a.clone() * b.clone();
//         let e = a.clone() + b.clone();
//         let f = d * e;
//         f.backward();
//         println!("{}", f);
//
//         assert_eq!(a.grad(), -3.0);
//         assert_eq!(b.grad(), -8.0);
//     }
//
//     #[test]
//     fn test_exp_div_sub_equivalence_to_tanh() {
//         let x1 = Tensor::new(2.0);
//         let x2 = Tensor::new(0.0);
//         let w1 = Tensor::new(-3.0);
//         let w2 = Tensor::new(1.0);
//         let b = Tensor::new(6.881_373_4);
//
//         let w1x1 = x1.clone() * w1;
//         let w2x2 = x2.clone() * w2;
//         let w1x1w2x2 = w1x1 + w2x2;
//         let n = w1x1w2x2 + b;
//         let o = n.tanh();
//         o.backward();
//
//         let x1_1 = Tensor::new(2.0);
//         let x2_1 = Tensor::new(0.0);
//         let w1_1 = Tensor::new(-3.0);
//         let w2_1 = Tensor::new(1.0);
//         let b_1 = Tensor::new(6.881_373_4);
//
//         let w1x1_1 = x1_1.clone() * w1_1;
//         let w2x2_1 = x2_1.clone() * w2_1;
//         let w1x1w2x2_1 = w1x1_1 + w2x2_1;
//         let n_1 = w1x1w2x2_1 + b_1;
//         let e_1 = (n_1 * 2.0).exp();
//         let o_1 = (e_1.clone() - 1.0) / (e_1 + 1.0);
//         o_1.backward();
//
//         let tol = 1e-6;
//         assert!((o.data() - o_1.data()).abs() < tol);
//         assert!((x1.grad() - x1_1.grad()).abs() < tol);
//         assert!((x2.grad() - x2_1.grad()).abs() < tol);
//     }
//
//     #[test]
//     fn test_sanity_check() {
//         let x = Tensor::new(-4.0);
//         let z = Tensor::new(2.0) * x.clone() + Tensor::new(2.0) + x.clone();
//         let q = z.clone().relu() + z.clone() * x.clone();
//         let h = (z.clone() * z.clone()).relu();
//         let y = h + q.clone() + q * x.clone();
//         y.backward();
//
//         // These are the correct values from the PyTorch reference
//         let y_expected = -20.0;
//         let x_grad_expected = 46.0;
//
//         let tol = 1e-6;
//         assert!((y.data() - y_expected).abs() < tol);
//         assert!((x.grad() - x_grad_expected).abs() < tol);
//     }
//
//     #[test]
//     fn test_more_ops() {
//         let a = Tensor::new(-4.0);
//         let b = Tensor::new(2.0);
//
//         let mut c = a.clone() + b.clone();
//         let mut d = a.clone() * b.clone() + b.clone().powf(3.0);
//
//         // Using `c = c + ...` to correctly build the graph
//         c = c.clone() + c.clone() + Tensor::new(1.0);
//         c = c.clone() + Tensor::new(1.0) + c.clone() + (-a.clone());
//
//         d = d.clone() + d.clone() * Tensor::new(2.0) + (b.clone() + a.clone()).relu();
//         d = d.clone() + Tensor::new(3.0) * d.clone() + (b.clone() - a.clone()).relu();
//
//         let e = c - d;
//         let f = e.clone().powf(2.0);
//         let mut g = f.clone() / 2.0;
//         g = g + Tensor::new(10.0) / f;
//         g.backward();
//
//         // These are the correct values from the PyTorch reference
//         let g_expected = 24.704_082;
//         let a_grad_expected = 138.833_82;
//         let b_grad_expected = 645.577_3;
//
//         let tol = 1e-6;
//         assert!((g.data() - g_expected).abs() < tol);
//         assert!((a.grad() - a_grad_expected).abs() < tol);
//         assert!((b.grad() - b_grad_expected).abs() < tol);
//     }
// }
