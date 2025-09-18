use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Display},
    ops::{Add, Div, Mul},
    rc::Rc,
};

use ordered_float::NotNan;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
enum Op {
    Add { lhs: Value, rhs: Value },
    Mul { lhs: Value, rhs: Value },
    Div { lhs: Value, rhs: Value },
    Tanh { input: Value },
}

impl Op {
    fn inputs(&self) -> impl Iterator<Item = &Value> {
        match self {
            Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } | Op::Div { lhs, rhs } => {
                vec![lhs, rhs].into_iter()
            }
            Op::Tanh { input } => vec![input].into_iter(),
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Add { .. } => write!(f, "+"),
            Op::Mul { .. } => write!(f, "*"),
            Op::Div { .. } => write!(f, "/"),
            Op::Tanh { .. } => write!(f, "tanh"),
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
struct Value {
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

    pub fn grad(&self) -> f32 {
        self.inner.borrow().grad.into_inner()
    }

    pub fn op(&self) -> Option<Op> {
        self.inner.borrow().op.clone()
    }

    pub fn set_op(&self, op: Option<Op>) {
        self.inner.borrow_mut().op = op;
    }

    fn set_grad(&self, val: f32) {
        self.inner.borrow_mut().grad = NotNan::new(val).unwrap();
    }

    fn increase_grad(&self, val: f32) {
        self.inner.borrow_mut().grad += NotNan::new(val).unwrap();
    }

    fn _backward(&mut self) {
        // assume self.grad has been initialized
        if let Some(op) = &mut self.op() {
            match op {
                Op::Add { lhs, rhs } => {
                    // 1.0 is local gradient, times with self.grad() for chain rule
                    lhs.increase_grad(1.0 * self.grad());
                    rhs.increase_grad(1.0 * self.grad());
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
            }
        }
    }

    fn topo(&self) -> impl DoubleEndedIterator<Item = Value> {
        // TODO: perf: less clones
        let mut topo = vec![];
        #[allow(
            clippy::mutable_key_type,
            reason = "Using identity of Value for visited set"
        )]
        let mut visited = HashSet::new();
        let mut stack = vec![self.clone()];
        while let Some(node) = stack.pop() {
            if visited.insert(node.clone()) {
                if let Some(op) = node.op() {
                    stack.extend(op.inputs().cloned())
                }
                topo.push(node);
            }
        }
        topo.into_iter()
    }

    pub fn backward(&mut self) {
        // topo sort to ensure that when we compute gradient for a node, all gradients of its children have
        // already been computed
        // mark the top of the computation graph
        self.set_grad(1.0);
        self.topo().for_each(|mut node| {
            node._backward();
        });
    }
}

trait Tanh {
    fn tanh(self) -> Self;
}

impl Tanh for Value {
    fn tanh(self) -> Self {
        let res = Self::new(((self.data() * 2.0).exp() - 1.0) / ((self.data() * 2.0).exp() + 1.0));
        res.set_op(Op::Tanh { input: self }.into());
        res
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
                    Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } | Op::Div { lhs, rhs } => {
                        fmt_with_indent(lhs, f, indent + 1)?;
                        fmt_with_indent(rhs, f, indent + 1)?;
                    }
                    Op::Tanh { input } => {
                        fmt_with_indent(input, f, indent + 1)?;
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

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let res = Self::new(self.data() * rhs.data());
        res.set_op(Op::Mul { lhs: self, rhs }.into());
        res
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

fn main() {
    // inputs
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    // weights
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    // bias
    let b = Value::new(6.881_373_4);
    // hidden layers
    let w1x1 = x1 * w1;
    let w2x2 = x2 * w2;
    let w1x1w2x2 = w1x1 + w2x2;
    let n = w1x1w2x2 + b;
    let mut o = n.tanh();
    o.backward();
    println!("{}", o);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_single_add() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let mut c = a.clone() + b.clone();
        c.backward();

        assert_eq!(c.grad(), 1.0);

        if let Some(Op::Add { lhs, rhs }) = c.op() {
            assert_eq!(lhs.grad(), 1.0);
            assert_eq!(rhs.grad(), 1.0);
        } else {
            panic!("expected add operation");
        }
    }

    #[test]
    fn test_backward_single_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let mut c = a.clone() * b.clone();
        c.backward();

        assert_eq!(c.grad(), 1.0);

        if let Some(Op::Mul { lhs, rhs }) = c.op() {
            assert_eq!(lhs.grad(), 3.0); // d(ab)/da = b
            assert_eq!(rhs.grad(), 2.0); // d(ab)/db = a
        } else {
            panic!("expected mul operation");
        }
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
        let mut result = abc.clone() * d.clone(); // 20
        result.backward();

        assert_eq!(result.data(), 20.0);
        assert_eq!(result.grad(), 1.0);

        if let Some(Op::Mul {
            lhs: abc_val,
            rhs: d_val,
        }) = result.op()
        {
            assert_eq!(abc_val.grad(), 4.0);
            assert_eq!(d_val.grad(), 5.0);

            if let Some(Op::Add {
                lhs: ab_val,
                rhs: c_val,
            }) = abc_val.op()
            {
                assert_eq!(ab_val.grad(), 4.0);
                assert_eq!(c_val.grad(), 4.0);

                if let Some(Op::Mul {
                    lhs: a_val,
                    rhs: b_val,
                }) = ab_val.op()
                {
                    assert_eq!(a_val.grad(), 8.0); // chain rule: 4 * b
                    assert_eq!(b_val.grad(), 4.0); // chain rule: 4 * a
                } else {
                    panic!("expected mul operation for ab");
                }
            } else {
                panic!("expected add operation for abc");
            }
        } else {
            panic!("expected mul operation for result");
        }
    }

    #[test]
    fn test_multiple_operations() {
        // a + b * c
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = Value::new(3.0);

        let bc = b.clone() * c.clone(); // 6
        let mut result = a.clone() + bc.clone(); // 7
        result.backward();

        assert_eq!(result.data(), 7.0);
        assert_eq!(result.grad(), 1.0);

        if let Some(Op::Add {
            lhs: a_val,
            rhs: bc_val,
        }) = result.op()
        {
            assert_eq!(a_val.grad(), 1.0);
            assert_eq!(bc_val.grad(), 1.0);

            if let Some(Op::Mul {
                lhs: b_val,
                rhs: c_val,
            }) = bc_val.op()
            {
                assert_eq!(b_val.grad(), 3.0);
                assert_eq!(c_val.grad(), 2.0);
            } else {
                panic!("expected mul operation for bc");
            }
        } else {
            panic!("expected add operation for result");
        }
    }
}
