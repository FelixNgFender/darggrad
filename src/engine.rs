use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Display},
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use ordered_float::NotNan;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
enum Op {
    Add { lhs: Value, rhs: Value },
    Neg { input: Value },
    Sub { lhs: Value, rhs: Value },
    Mul { lhs: Value, rhs: Value },
    Div { lhs: Value, rhs: Value },
    Tanh { input: Value },
    Relu { input: Value },
    Exp { exp: Value },
    Powf { base: Value, exp: Value },
}

impl Op {
    fn inputs(&self) -> impl Iterator<Item = &Value> {
        match self {
            Op::Add { lhs, rhs }
            | Op::Sub { lhs, rhs }
            | Op::Mul { lhs, rhs }
            | Op::Div { lhs, rhs } => vec![lhs, rhs].into_iter(),
            Op::Tanh { input } | Op::Relu { input } | Op::Neg { input } => vec![input].into_iter(),
            Op::Exp { exp } => vec![exp].into_iter(),
            Op::Powf { base, exp } => vec![base, exp].into_iter(),
        }
    }
}

impl Display for Op {
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

#[derive(PartialEq, Eq, Hash, Debug)]
struct ValueInner {
    op: Option<Op>,
    data: NotNan<f32>,
    /// gradient relative to the first caller of `backward()`
    grad: NotNan<f32>,
}

#[derive(Clone, Debug)]
pub struct Value {
    /// allows a `Value` to be shared and mutated across multiple owners, essentially allowing
    /// construction of a DAG computation graph (a `Value` may contribute to more than 1 output).
    inner: Rc<RefCell<ValueInner>>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // hash the pointer address instead of the inner mutating contents
        // for autodiff, we don’t care about value-equality, only care about node identity.
        Rc::as_ptr(&self.inner).hash(state);
    }
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                data: data.try_into().unwrap(),
                op: None,
                grad: NotNan::new(0.0).unwrap(),
            })),
        }
    }

    pub fn data(&self) -> f32 {
        self.inner.borrow().data.into_inner()
    }

    pub fn increase_data(&self, val: f32) {
        self.inner.borrow_mut().data += NotNan::new(val).unwrap();
    }

    pub fn grad(&self) -> f32 {
        self.inner.borrow().grad.into_inner()
    }

    fn op(&self) -> Option<Op> {
        self.inner.borrow().op.clone()
    }

    fn set_op(&self, op: Option<Op>) {
        self.inner.borrow_mut().op = op;
    }

    pub fn set_grad(&self, val: f32) {
        self.inner.borrow_mut().grad = NotNan::new(val).unwrap();
    }

    fn increase_grad(&self, val: f32) {
        self.inner.borrow_mut().grad += NotNan::new(val).unwrap();
    }

    fn _backward(&self) {
        // assume self.grad has been initialized
        if let Some(op) = &self.op() {
            match op {
                Op::Add { lhs, rhs } => {
                    // 1.0 is local gradient, times with self.grad() for chain rule
                    lhs.increase_grad(1.0 * self.grad());
                    rhs.increase_grad(1.0 * self.grad());
                }
                Op::Neg { input } => {
                    input.increase_grad(-1.0 * self.grad());
                }
                Op::Sub { lhs, rhs } => {
                    lhs.increase_grad(1.0 * self.grad());
                    rhs.increase_grad(-1.0 * self.grad());
                }
                Op::Mul { lhs, rhs } => {
                    lhs.increase_grad(rhs.data() * self.grad());
                    rhs.increase_grad(lhs.data() * self.grad());
                }
                Op::Div { lhs, rhs } => {
                    // c = a/b
                    // dc/da = 1/b
                    // dc/db = -a/(b^2)
                    lhs.increase_grad((1.0 / rhs.data()) * self.grad());
                    rhs.increase_grad((-lhs.data() / rhs.data().powi(2)) * self.grad());
                }
                Op::Tanh { input } => {
                    // y = tanhx
                    // dy/dx = 1 - y^2
                    input.increase_grad((1.0 - self.data().powi(2)) * self.grad());
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

    fn topo(&self) -> impl DoubleEndedIterator<Item = Value> {
        #[allow(
            clippy::mutable_key_type,
            reason = "Using identity of Value for visited set"
        )]
        fn build_topo(node: &Value, visited: &mut HashSet<Value>, topo: &mut Vec<Value>) {
            if visited.insert(node.clone()) {
                if let Some(op) = node.op() {
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
            reason = "Using identity of Value for visited set"
        )]
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);
        topo.into_iter()
    }

    pub fn backward(&self) {
        // mark the top of the computation graph
        self.set_grad(1.0);
        // topo sort to ensure that when we compute gradient for a node, all gradients of its children have
        // already been computed
        self.topo().rev().for_each(|node| {
            node._backward();
        });
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_with_indent(
            value: &Value,
            f: &mut fmt::Formatter<'_>,
            indent: usize,
        ) -> fmt::Result {
            for _ in 0..indent {
                write!(f, "    ")?; // four spaces per indent level
            }

            match &value.op() {
                Some(op) => writeln!(
                    f,
                    "Value(data={:.4}, grad={:.4}, op={})",
                    value.data(),
                    value.grad(),
                    op
                )?,
                None => writeln!(
                    f,
                    "Value(data={:.4}, grad={:.4})",
                    value.data(),
                    value.grad()
                )?,
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

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = Self::new(self.data() + rhs.data());
        res.set_op(Op::Add { lhs: self, rhs }.into());
        res
    }
}

impl Add<f32> for Value {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        self + Value::new(rhs)
    }
}

impl Add<Value> for f32 {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value::new(self) + rhs
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, x| acc + x)
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let res = Self::new(-self.data());
        res.set_op(Op::Neg { input: self }.into());
        res
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let res = Self::new(self.data() - rhs.data());
        res.set_op(Op::Sub { lhs: self, rhs }.into());
        res
    }
}

impl Sub<f32> for Value {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        self - Value::new(rhs)
    }
}

impl Sub<Value> for f32 {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        Value::new(self) - rhs
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let res = Self::new(self.data() * rhs.data());
        res.set_op(Op::Mul { lhs: self, rhs }.into());
        res
    }
}

impl Mul<f32> for Value {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self * Value::new(rhs)
    }
}

impl Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value::new(self) * rhs
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let res = Self::new(self.data() / rhs.data());
        res.set_op(Op::Div { lhs: self, rhs }.into());
        res
    }
}

impl Div<f32> for Value {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self / Value::new(rhs)
    }
}

impl Div<Value> for f32 {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        Value::new(self) / rhs
    }
}

pub trait Tanh {
    fn tanh(self) -> Self;
}

impl Tanh for Value {
    fn tanh(self) -> Self {
        let res = Self::new(((self.data() * 2.0).exp() - 1.0) / ((self.data() * 2.0).exp() + 1.0));
        res.set_op(Op::Tanh { input: self }.into());
        res
    }
}

trait Relu {
    fn relu(self) -> Self;
}

impl Relu for Value {
    fn relu(self) -> Self {
        let res = Self::new(self.data().max(0.0));
        res.set_op(Op::Relu { input: self }.into());
        res
    }
}

trait Exp {
    type Output;

    fn exp(self) -> Self::Output;
}

impl Exp for Value {
    type Output = Value;

    fn exp(self) -> Self::Output {
        let res = Self::new(self.data().exp());
        res.set_op(Op::Exp { exp: self }.into());
        res
    }
}

pub trait Powf {
    type Output;

    fn powf(self, exp: f32) -> Self::Output;
}

impl Powf for Value {
    type Output = Value;

    fn powf(self, exp: f32) -> Self::Output {
        let res = Self::new(self.data().powf(exp));
        res.set_op(
            Op::Powf {
                base: self,
                exp: Value::new(exp),
            }
            .into(),
        );
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_single_add() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() + b.clone();
        c.backward();

        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_backward_single_neg() {
        let a = Value::new(5.0);
        let b = -a.clone();
        b.backward();

        assert_eq!(b.data(), -5.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(a.grad(), -1.0);
    }

    #[test]
    fn test_backward_single_sub() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() - b.clone();
        c.backward();

        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), -1.0);
    }

    #[test]
    fn test_backward_single_exp() {
        let a = Value::new(2.0);
        let b = a.clone().exp();
        b.backward();

        let tol = 1e-6;
        assert!((b.data() - 2.0_f32.exp()).abs() < tol);
        assert_eq!(b.grad(), 1.0);
        assert!((a.grad() - 2.0_f32.exp()).abs() < tol); // d/dx(e^x) = e^x
    }

    #[test]
    fn test_backward_single_square() {
        let a = Value::new(3.0);
        let b = a.clone().powf(2.0);
        b.backward();

        assert_eq!(b.data(), 9.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(a.grad(), 6.0);
    }

    #[test]
    fn test_square_negative_base() {
        let a = Value::new(-2.0);
        let b = a.clone().powf(2.0);
        b.backward();

        assert_eq!(b.data(), 4.0);
        assert_eq!(a.grad(), -4.0);
    }

    #[test]
    fn test_backward_single_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() * b.clone();
        c.backward();

        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 3.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn test_chain_rule_computation() {
        // ((a * b) + c) * d
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = Value::new(3.0);
        let d = Value::new(4.0);

        let ab = a.clone() * b.clone(); // 2
        let abc = ab.clone() + c.clone(); // 5
        let result = abc.clone() * d.clone(); // 20
        result.backward();

        assert_eq!(result.data(), 20.0);
        assert_eq!(result.grad(), 1.0);
        assert_eq!(abc.grad(), 4.0);
        assert_eq!(d.grad(), 5.0);
        assert_eq!(ab.grad(), 4.0);
        assert_eq!(c.grad(), 4.0);
        assert_eq!(a.grad(), 8.0); // chain rule: 4 * b
        assert_eq!(b.grad(), 4.0); // chain rule: 4 * a
    }

    #[test]
    fn test_multiple_operations() {
        // a + b * c
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = Value::new(3.0);

        let bc = b.clone() * c.clone(); // 6
        let result = a.clone() + bc.clone(); // 7
        result.backward();

        assert_eq!(result.data(), 7.0);
        assert_eq!(result.grad(), 1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(bc.grad(), 1.0);
        assert_eq!(b.grad(), 3.0);
        assert_eq!(c.grad(), 2.0);
    }

    #[test]
    fn test_square_chain_rule() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let ab = a.clone() * b.clone(); // 6
        let result = ab.clone().powf(2.0); // 36
        result.backward();

        assert_eq!(result.data(), 36.0);
        assert_eq!(ab.grad(), 12.0); // 2 * 6 = 12
        assert_eq!(a.grad(), 36.0); // 12 * 3 = 36
        assert_eq!(b.grad(), 24.0); // 12 * 2 = 24
    }

    #[test]
    fn test_shared_node() {
        let a = Value::new(3.0);
        let b = a.clone() + a.clone();
        b.backward();

        assert_eq!(a.grad(), 2.0);
    }

    #[test]
    fn test_shared_node_complex() {
        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let d = a.clone() * b.clone();
        let e = a.clone() + b.clone();
        let f = d * e;
        f.backward();
        println!("{}", f);

        assert_eq!(a.grad(), -3.0);
        assert_eq!(b.grad(), -8.0);
    }

    #[test]
    fn test_exp_div_sub_equivalence_to_tanh() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.881_373_4);

        let w1x1 = x1.clone() * w1;
        let w2x2 = x2.clone() * w2;
        let w1x1w2x2 = w1x1 + w2x2;
        let n = w1x1w2x2 + b;
        let o = n.tanh();
        o.backward();

        let x1_1 = Value::new(2.0);
        let x2_1 = Value::new(0.0);
        let w1_1 = Value::new(-3.0);
        let w2_1 = Value::new(1.0);
        let b_1 = Value::new(6.881_373_4);

        let w1x1_1 = x1_1.clone() * w1_1;
        let w2x2_1 = x2_1.clone() * w2_1;
        let w1x1w2x2_1 = w1x1_1 + w2x2_1;
        let n_1 = w1x1w2x2_1 + b_1;
        let e_1 = (n_1 * 2.0).exp();
        let o_1 = (e_1.clone() - 1.0) / (e_1 + 1.0);
        o_1.backward();

        let tol = 1e-6;
        assert!((o.data() - o_1.data()).abs() < tol);
        assert!((x1.grad() - x1_1.grad()).abs() < tol);
        assert!((x2.grad() - x2_1.grad()).abs() < tol);
    }

    #[test]
    fn test_sanity_check() {
        let x = Value::new(-4.0);
        let z = Value::new(2.0) * x.clone() + Value::new(2.0) + x.clone();
        let q = z.clone().relu() + z.clone() * x.clone();
        let h = (z.clone() * z.clone()).relu();
        let y = h + q.clone() + q * x.clone();
        y.backward();

        // These are the correct values from the PyTorch reference
        let y_expected = -20.0;
        let x_grad_expected = 46.0;

        let tol = 1e-6;
        assert!((y.data() - y_expected).abs() < tol);
        assert!((x.grad() - x_grad_expected).abs() < tol);
    }

    #[test]
    fn test_more_ops() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);

        let mut c = a.clone() + b.clone();
        let mut d = a.clone() * b.clone() + b.clone().powf(3.0);

        // Using `c = c + ...` to correctly build the graph
        c = c.clone() + c.clone() + Value::new(1.0);
        c = c.clone() + Value::new(1.0) + c.clone() + (-a.clone());

        d = d.clone() + d.clone() * Value::new(2.0) + (b.clone() + a.clone()).relu();
        d = d.clone() + Value::new(3.0) * d.clone() + (b.clone() - a.clone()).relu();

        let e = c - d;
        let f = e.clone().powf(2.0);
        let mut g = f.clone() / 2.0;
        g = g + Value::new(10.0) / f;
        g.backward();

        // These are the correct values from the PyTorch reference
        let g_expected = 24.704_082;
        let a_grad_expected = 138.833_82;
        let b_grad_expected = 645.577_3;

        let tol = 1e-6;
        assert!((g.data() - g_expected).abs() < tol);
        assert!((a.grad() - a_grad_expected).abs() < tol);
        assert!((b.grad() - b_grad_expected).abs() < tol);
    }
}
