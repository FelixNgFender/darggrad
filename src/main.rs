use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Display},
    ops::{Add, Div, Mul},
    rc::Rc,
};

use ordered_float::NotNan;

#[derive(PartialEq, Eq, Hash)]
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

    fn inputs_mut(&mut self) -> impl Iterator<Item = &mut Value> {
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

#[derive(PartialEq, Eq, Clone)]
struct Value {
    /// Allows a `Value` to be shared and mutated across multiple owners, essentially allowing
    /// construction of a DAG computation graph (a `Value` may contribute to more than 1 output).
    inner: Rc<RefCell<ValueInner>>,
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.borrow().hash(state);
    }
}

#[derive(PartialEq, Eq, Hash)]
struct ValueInner {
    op: Option<Op>,
    data: NotNan<f32>,
    /// gradient relative to the first caller of `backward()`
    grad: NotNan<f32>,
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

    fn _backward(&mut self) {
        // assume self.grad has been initialized
        if let Some(op) = &mut self.inner.borrow_mut().op {
            match op {
                Op::Add { lhs, rhs } => {
                    // local gradients only
                    // x
                    lhs.inner.borrow_mut().grad += NotNan::new(1.0).unwrap();
                    // y
                    rhs.inner.borrow_mut().grad += NotNan::new(1.0).unwrap();
                }
                Op::Mul { lhs, rhs } => {
                    lhs.inner.borrow_mut().grad += rhs.inner.borrow().data;
                    rhs.inner.borrow_mut().grad += lhs.inner.borrow().data;
                }
                Op::Div { lhs, rhs } => {
                    // c = a/b
                    // dc/da = 1/b
                    // dc/db = -a/(b^2)
                    lhs.inner.borrow_mut().grad +=
                        NotNan::new(1.0).unwrap() / rhs.inner.borrow().data;
                    rhs.inner.borrow_mut().grad +=
                        NotNan::new(-lhs.inner.borrow().data / (rhs.inner.borrow().data.powi(2)))
                            .unwrap();
                }
                Op::Tanh { input } => {
                    // y = tanhx
                    // dy/dx = 1 - y^2
                    input.inner.borrow_mut().grad +=
                        NotNan::new(1.0 - self.inner.borrow().data.powi(2)).unwrap();
                }
            }
            op.inputs_mut().for_each(|input| {
                // chain rule
                input.inner.borrow_mut().grad *= self.inner.borrow().grad;
            });
        }
    }

    fn topo(&self) -> impl DoubleEndedIterator<Item = Value> {
        // TODO: perf: less clones
        let mut topo = vec![];
        let mut visited = HashSet::new();
        let mut stack = vec![self.clone()];
        while let Some(node) = stack.pop() {
            if !visited.contains(&node) {
                visited.insert(node.clone());
                if let Some(op) = &node.inner.borrow().op {
                    for input in op.inputs() {
                        stack.push(input.clone());
                    }
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
        self.inner.borrow_mut().grad = NotNan::new(1.0).unwrap();
        self.topo().rev().for_each(|mut node| {
            node._backward();
        });
    }

    fn tanh(self) -> Self {
        let mut res = Self::new(((2.0 * self.data).exp() - 1.0) / ((2.0 * self.data).exp() + 1.0));
        res.op = Op::Tanh {
            input: Rc::new(RefCell::new(self)),
        }
        .into();
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

            let innner = value.inner.borrow();
            match inner.op {
                Some(op) => writeln!(
                    f,
                    "Value(data={:.4}, grad={:.4}, op={})",
                    value.data, value.grad, op
                )?,
                None => writeln!(f, "Value(data={:.4}, grad={:.4})", value.data, value.grad)?,
            }

            if let Some(op) = &value.op {
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
        let mut res = Self::new(self.data + rhs.data);
        res.op = Op::Add {
            lhs: Rc::new(RefCell::new(self)),
            rhs: Rc::new(RefCell::new(rhs)),
        }
        .into();
        res
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Self::new(self.data * rhs.data);
        res.op = Op::Mul {
            lhs: Rc::new(RefCell::new(self)),
            rhs: Rc::new(RefCell::new(rhs)),
        }
        .into();
        res
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut res = Self::new(self.data / rhs.data);
        res.op = Op::Div {
            lhs: Rc::new(RefCell::new(self)),
            rhs: Rc::new(RefCell::new(rhs)),
        }
        .into();
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
    let b = Value::new(6.881_373_587_019_543_2);
    // hidden layers
    let w1x1 = x1.clone() * w1;
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
        let mut c = a + b;
        c.backward();

        assert_eq!(c.grad, 1.0);

        if let Some(op) = &c.op {
            match &**op {
                Op::Add { lhs, rhs } => {
                    assert_eq!(lhs.borrow().grad, 1.0);
                    assert_eq!(rhs.grad, 1.0);
                }
                _ => panic!("expected add operation"),
            }
        }
    }

    #[test]
    fn test_backward_single_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let mut c = a * b;
        c.backward();

        assert_eq!(c.grad, 1.0);

        if let Some(op) = &c.op {
            match &**op {
                Op::Mul { lhs, rhs } => {
                    assert_eq!(lhs.grad, 3.0); // gradient is rhs.data
                    assert_eq!(rhs.grad, 2.0); // gradient is lhs.data
                }
                _ => panic!("expected mul operation"),
            }
        }
    }

    #[test]
    fn test_chain_rule_computation() {
        // Test: ((a * b) + c) * d
        // Where we can verify chain rule application
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = Value::new(3.0);
        let d = Value::new(4.0);

        let ab = a * b; // ab = 2
        let abc = ab + c; // abc = 5  
        let mut result = abc * d; // result = 20
        result.backward();

        assert_eq!(result.data, 20.0);
        assert_eq!(result.grad, 1.0);

        // Verify gradients
        if let Some(op) = &result.op {
            match &**op {
                Op::Mul {
                    lhs: abc_val,
                    rhs: d_val,
                } => {
                    assert_eq!(abc_val.grad, 4.0); // d(result)/d(abc) = d = 4
                    assert_eq!(d_val.grad, 5.0); // d(result)/d(d) = abc = 5

                    // Check abc gradients
                    if let Some(abc_op) = &abc_val.op {
                        match &**abc_op {
                            Op::Add {
                                lhs: ab_val,
                                rhs: c_val,
                            } => {
                                assert_eq!(ab_val.grad, 4.0); // chain rule: 4 * 1
                                assert_eq!(c_val.grad, 4.0); // chain rule: 4 * 1

                                // Check ab gradients
                                if let Some(ab_op) = &ab_val.op {
                                    match &**ab_op {
                                        Op::Mul {
                                            lhs: a_val,
                                            rhs: b_val,
                                        } => {
                                            assert_eq!(a_val.grad, 8.0); // chain rule: 4 * b = 4 * 2
                                            assert_eq!(b_val.grad, 4.0); // chain rule: 4 * a = 4 * 1
                                        }
                                        _ => panic!("expected mul operation for ab"),
                                    }
                                }
                            }
                            _ => panic!("expected add operation for abc"),
                        }
                    }
                }
                _ => panic!("expected mul operation for result"),
            }
        }
    }

    #[test]
    fn test_multiple_operations() {
        // Test: a + b * c
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = Value::new(3.0);

        let bc = b * c; // bc = 6
        let mut result = a + bc; // result = 7
        result.backward();

        assert_eq!(result.data, 7.0);
        assert_eq!(result.grad, 1.0);

        if let Some(op) = &result.op {
            match &**op {
                Op::Add {
                    lhs: a_val,
                    rhs: bc_val,
                } => {
                    assert_eq!(a_val.grad, 1.0); // d(result)/da = 1
                    assert_eq!(bc_val.grad, 1.0); // d(result)/d(bc) = 1

                    if let Some(bc_op) = &bc_val.op {
                        match &**bc_op {
                            Op::Mul {
                                lhs: b_val,
                                rhs: c_val,
                            } => {
                                assert_eq!(b_val.grad, 3.0); // chain rule: 1 * c = 1 * 3
                                assert_eq!(c_val.grad, 2.0); // chain rule: 1 * b = 1 * 2
                            }
                            _ => panic!("expected mul operation for bc"),
                        }
                    }
                }
                _ => panic!("expected add operation for result"),
            }
        }
    }
}
