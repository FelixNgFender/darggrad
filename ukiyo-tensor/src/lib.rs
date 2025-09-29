use std::{
    fmt::{Debug, Display},
    ops::{Index, IndexMut},
};

use num_traits::Zero;

#[derive(Debug)]
struct TensorShape {
    /// An n-tuple that represents the length of each dimension in the tensor.
    shape: Vec<usize>,
    /// Pre-computed strides for each dimension.
    strides: Vec<usize>,
}

impl TensorShape {
    fn new(shape: &[usize]) -> Self {
        let mut strides = vec![1; shape.len()];
        (0..shape.len().saturating_sub(1)).rev().for_each(|i| {
            strides[i] = strides[i + 1] * shape[i + 1];
        });
        TensorShape {
            shape: shape.to_vec(),
            strides,
        }
    }

    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Converts n-dimensional indices to a linear index.
    fn ravel_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Indices length must match tensor shape dimensions"
        );

        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Converts a linear index to n-dimensional indices.
    fn unravel_index(&self, index: usize) -> Vec<usize> {
        if self.shape.is_empty() {
            return vec![];
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remaining_index = index;
        for (i, &stride) in self.strides.iter().enumerate() {
            indices[i] = remaining_index / stride;
            remaining_index %= stride;
        }
        indices
    }

    /// Permutes the dimensions of the tensor shape according to the given axes.
    fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            self.shape.len(),
            "Axes length must match tensor shape dimensions"
        );
        let shape = axes.iter().map(|&i| self.shape[i]).collect();
        let strides = axes.iter().map(|&i| self.strides[i]).collect();
        Self { shape, strides }
    }
}

impl Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorShape(shape={:?}, strides={:?})",
            self.shape, self.strides
        )
    }
}

#[derive(Debug, Clone)]
struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T> TensorStorage<T>
where
    T: Zero + Clone,
{
    fn zeros(size: usize) -> Self {
        Self {
            data: vec![T::zero(); size],
        }
    }
}

impl<T> Index<usize> for TensorStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for TensorStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[derive(Debug)]
pub struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T> Tensor<T>
where
    T: Zero + Clone,
{
    /// Creates a tensor filled with zeros given a shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let shape = TensorShape::new(shape);
        let storage = TensorStorage::<T>::zeros(shape.size());
        Self { shape, storage }
    }
}

impl<T: Clone> Tensor<T> {
    fn permute(&self, axes: &[usize]) -> Self {
        Self {
            shape: self.shape.permute(axes),
            storage: self.storage.clone(), // PERF:
        }
    }
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.storage[self.shape.ravel_index(indices)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.storage[self.shape.ravel_index(indices)]
    }
}

impl<T: Debug> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tensor(shape={}):", self.shape)?;

        // First pass: find max element width (as string)
        let widths: Vec<usize> = (0..self.shape.size())
            .map(|i| format!("{:?}", self.storage[i]).len())
            .collect();
        let max_width = widths.iter().copied().max().unwrap_or(1);

        fn format_recursive<T: Debug>(
            f: &mut std::fmt::Formatter<'_>,
            shape: &TensorShape,
            storage: &TensorStorage<T>,
            dim: usize,
            offset: usize,
            indent: usize,
            max_width: usize,
        ) -> std::fmt::Result {
            if dim == shape.shape.len() - 1 {
                write!(f, "{:indent$}[", "", indent = indent)?;
                for i in 0..shape.shape[dim] {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    let idx = offset + i * shape.strides[dim];
                    write!(
                        f,
                        "{:>width$}",
                        format!("{:?}", storage[idx]),
                        width = max_width
                    )?;
                }
                write!(f, "]")
            } else {
                writeln!(f, "{:indent$}[", "", indent = indent)?;
                for i in 0..shape.shape[dim] {
                    if i > 0 {
                        writeln!(f, ",")?;
                    }
                    let idx = offset + i * shape.strides[dim];
                    format_recursive(f, shape, storage, dim + 1, idx, indent + 2, max_width)?;
                }
                writeln!(f)?;
                write!(f, "{:indent$}]", "", indent = indent)
            }
        }

        format_recursive(f, &self.shape, &self.storage, 0, 0, 0, max_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
