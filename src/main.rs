use std::{
    fmt::{self, Display},
    ops::{Add, Mul},
};

#[derive(Clone, Debug)]
enum Op {
    Add { lhs: Value, rhs: Value },
    Mul { lhs: Value, rhs: Value },
}

impl Op {
    fn children(&mut self) -> impl Iterator<Item = &mut Value> {
        match self {
            Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } => vec![lhs, rhs].into_iter(),
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Add { .. } => write!(f, "+"),
            Op::Mul { .. } => write!(f, "*"),
        }
    }
}

#[derive(Clone, Debug)]
struct Value {
    data: f32,
    op: Option<Box<Op>>,
    // gradient relative to the first caller of `backward()`
    grad: f32,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Value {
            data,
            op: None,
            grad: 0.0,
        }
    }

    fn _backward(&mut self) {
        // assume self.grad has been initialized
        if let Some(op) = &mut self.op {
            match op.as_mut() {
                Op::Add { lhs, rhs } => {
                    // local gradients only
                    lhs.grad = 1.0;
                    rhs.grad = 1.0;
                }
                Op::Mul { lhs, rhs } => {
                    lhs.grad = rhs.data;
                    rhs.grad = lhs.data;
                }
            }
            op.children().for_each(|child: &mut Value| {
                // chain rule
                child.grad *= self.grad;
                child._backward();
            });
        }
    }

    pub fn backward(&mut self) {
        // mark the top of the computation graph
        self.grad = 1.0;
        self._backward();
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

            match &value.op {
                Some(op) => writeln!(
                    f,
                    "Value(data={:.4}, grad={:.4}, op={})",
                    value.data, value.grad, op
                )?,
                None => writeln!(f, "Value(data={:.4}, grad={:.4})", value.data, value.grad)?,
            }

            if let Some(op) = &value.op {
                match &**op {
                    Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } => {
                        fmt_with_indent(lhs, f, indent + 1)?;
                        fmt_with_indent(rhs, f, indent + 1)?;
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
        res.op = Box::new(Op::Add { lhs: self, rhs }).into();
        res
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Self::new(self.data * rhs.data);
        res.op = Box::new(Op::Mul { lhs: self, rhs }).into();
        res
    }
}

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let e = a * b;
    let d = e + c;
    let f = Value::new(-2.0);
    let mut l = d * f;
    l.backward();
    // dl/df = d = 4
    // dl/dd = f = -2
    //
    // dl/de = dl/dd * dd/de = -2 * 1 = -2
    // dl/dc = dl/dd * dd/dc = -2 * 1 = -2
    //
    // dl/da = dl/de * de/da = -2 * b = -2 * -3 = 6
    // dl/db = dl/de * de/db = -2 * a = -2 * 2 = -4
    //
    // Value(data=-8.0000, grad=1.0000, op=*)
    //     Value(data=4.0000, grad=-2.0000, op=+)
    //         Value(data=-6.0000, grad=-2.0000, op=*)
    //             Value(data=2.0000, grad=6.0000)
    //             Value(data=-3.0000, grad=-4.0000)
    //         Value(data=10.0000, grad=-2.0000)
    //     Value(data=-2.0000, grad=4.0000)
    println!("{}", l);
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
                    assert_eq!(lhs.grad, 1.0);
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
                            _ => panic!("Expected Mul operation for bc"),
                        }
                    }
                }
                _ => panic!("Expected Add operation for result"),
            }
        }
    }
}
