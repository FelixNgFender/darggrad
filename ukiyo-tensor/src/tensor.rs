use num_traits::{One, PrimInt, Zero};

use crate::{tensor_shape::TensorShape, tensor_storage::TensorStorage};

use std::{
    fmt::{Debug, Display},
    ops::{Index, IndexMut, RangeInclusive},
};
#[derive(Debug)]
pub struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        assert_eq!(
            shape.size(),
            data.len(),
            "Data length does not match tensor shape size"
        );
        let storage = TensorStorage { data };
        Self { shape, storage }
    }

    pub fn rand_with<D: rand_distr::Distribution<T>>(
        shape: Vec<usize>,
        generator: &mut impl rand::Rng,
        distr: D,
    ) -> Self {
        let shape = TensorShape::new(shape);
        let storage = TensorStorage {
            data: (0..shape.size()).map(|_| distr.sample(generator)).collect(),
        };
        Self { shape, storage }
    }

    /// Since storage is cloned in many operations, provide the "correct" way to view the tensor
    /// data according to its shape.
    pub fn data(&self) -> Vec<&T> {
        let mut res = vec![];
        let mut current_indices = vec![0; self.shape.dim()];
        let mut current_dim = current_indices.len() - 1;
        loop {
            let value = &self.storage.data[self.shape.ravel_index(&current_indices)];
            res.push(value);
            current_indices[current_dim] += 1;
            if current_indices[current_dim] >= self.shape.shape[current_dim] - 1 {
                if current_dim == 0 {
                    break;
                }
                current_dim -= 1;
            }
        }
        res
    }

    pub fn map<F, U>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&T) -> U,
    {
        Tensor {
            shape: self.shape.clone(), // PERF:
            storage: self.storage.map(f),
        }
    }
}

impl<T: PrimInt + Display> Tensor<T> {
    /// One-hot encodes an index tensor.
    ///
    /// - `indices`: a tensor of integer class indices.
    /// - `num_classes`: number of classes (must be > 0).
    ///
    /// Output shape = `indices.shape + [num_classes]`.
    pub fn one_hot(indices: &Tensor<T>, num_classes: usize) -> Self {
        assert!(num_classes > 0, "num_classes must be > 0");

        // New shape = indices.shape plus one more dimension
        let mut out_shape_vec = indices.shape.shape.clone();
        out_shape_vec.push(num_classes);
        let out_shape = TensorShape::new(out_shape_vec);

        let mut storage = TensorStorage::<T>::zeros(out_shape.size());

        // Loop through each index in the indices tensor
        for flat_idx in 0..indices.shape.size() {
            let mut multi_idx = indices.shape.unravel_index(flat_idx);
            let class_idx = indices.storage[flat_idx]
                .to_usize()
                .expect("Index out of bounds");
            assert!(
                class_idx < num_classes,
                "Index {} out of bounds for num_classes={}",
                class_idx,
                num_classes
            );

            // Position in output = multi_idx + [class_idx]
            multi_idx.push(class_idx);

            let out_flat_idx = out_shape.ravel_index(&multi_idx);
            storage[out_flat_idx] = T::one();
        }

        Tensor {
            shape: out_shape,
            storage,
        }
    }
}

impl<T: Clone> Tensor<T> {
    /// Permutes the dimensions of the tensor shape according to the given axes.
    pub fn permute(&self, axes: &[usize]) -> Self {
        Self {
            shape: self.shape.permute(axes),
            storage: self.storage.clone(), // PERF:
        }
    }

    /// Merges a range of dimensions into a single dimension.
    pub fn merge(&self, dim_range: RangeInclusive<usize>) -> Self {
        Self {
            shape: self.shape.merge(dim_range),
            storage: self.storage.clone(), // PERF:
        }
    }

    /// Splits a dimension into multiple dimensions according to the given shape.
    pub fn split(&self, dim: usize, shape: &[usize]) -> Self {
        Tensor {
            shape: self.shape.split(dim, shape),
            storage: self.storage.clone(), // PERF:
        }
    }

    /// Slices the tensor along a specified dimension using an inclusive range.
    pub fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Self {
        Tensor {
            shape: self.shape.slice(dim, range),
            storage: self.storage.clone(), // PERF:
        }
    }

    /// Skips elements in a specified dimension by a given step size.
    pub fn skip(&self, dim: usize, step: usize) -> Self {
        Tensor {
            shape: self.shape.skip(dim, step),
            storage: self.storage.clone(), // PERF:
        }
    }
}

impl<T: Zero + Clone> Tensor<T> {
    /// Creates a tensor filled with zeros given a shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        let storage = TensorStorage::<T>::zeros(shape.size());
        Self { shape, storage }
    }

    /// Reduces the tensor along a specified dimension using a binary function.
    ///
    /// If `keepdim` is true, the reduced dimension is retained with size 1.
    /// Else, the dimension is removed (i.e., squeezed).
    pub fn reduce<F>(&self, dim: usize, f: F, keepdim: bool) -> Tensor<T>
    where
        F: Fn(&T, &T) -> T,
    {
        assert!(
            dim < self.shape.shape.len(),
            "Dimension index out of bounds"
        );

        let result_storage = self.storage.reduce(&self.shape, dim, f, keepdim);

        // Calculate result shape by removing the reduction dimension
        let mut result_shape_vec = self.shape.shape.clone();
        if keepdim {
            result_shape_vec[dim] = 1;
        } else {
            result_shape_vec.remove(dim);
        }
        let result_tensor_shape = TensorShape::new(result_shape_vec);

        Tensor {
            shape: result_tensor_shape,
            storage: result_storage,
        }
    }

    /// Performs an element-wise binary operation between two tensors with broadcasting.
    pub fn broadcast_op<F, U, T2>(
        &self,
        other: &Tensor<T2>,
        corresponding_dimensions: &[(usize, usize)],
        f: F,
    ) -> Tensor<U>
    where
        F: Fn(&T, &T2) -> U,
        U: Zero + Clone,
    {
        let result_shape = self
            .shape
            .broadcast_shape(&other.shape, corresponding_dimensions);

        let result_storage = self.storage.broadcast_op(
            &self.shape,
            &other.storage,
            &other.shape,
            corresponding_dimensions,
            f,
        );

        Tensor {
            shape: result_shape,
            storage: result_storage,
        }
    }
}

impl<T: One + Clone> Tensor<T> {
    /// Creates a tensor filled with ones given a shape.
    pub fn ones(shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        let storage = TensorStorage {
            data: vec![T::one(); shape.size()],
        };
        Self { shape, storage }
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
            if shape.is_scalar() {
                write!(
                    f,
                    "{:>width$}",
                    format!("{:?}", storage[offset]),
                    width = max_width
                )
            } else if dim == shape.shape.len() - 1 {
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

    #[test]
    fn test_tensor_zeros_basic() {
        let tensor = Tensor::<f32>::zeros(vec![2, 3]);

        assert_eq!(tensor.shape.shape, vec![2, 3]);

        assert_eq!(tensor.storage.data.len(), 6);

        assert!(tensor.storage.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ravel_index() {
        let shape = TensorShape::new(vec![5]);

        assert_eq!(shape.ravel_index(&[0]), 0);
        assert_eq!(shape.ravel_index(&[1]), 1);
        assert_eq!(shape.ravel_index(&[2]), 2);
        assert_eq!(shape.ravel_index(&[3]), 3);
        assert_eq!(shape.ravel_index(&[4]), 4);

        let shape = TensorShape::new(vec![2, 3]);

        assert_eq!(shape.ravel_index(&[0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 2]), 2);
        assert_eq!(shape.ravel_index(&[1, 0]), 3);
        assert_eq!(shape.ravel_index(&[1, 1]), 4);
        assert_eq!(shape.ravel_index(&[1, 2]), 5);

        let shape = TensorShape::new(vec![2, 3, 4]);

        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 0, 2]), 2);
        assert_eq!(shape.ravel_index(&[0, 0, 3]), 3);
        assert_eq!(shape.ravel_index(&[0, 1, 0]), 4);
        assert_eq!(shape.ravel_index(&[0, 1, 1]), 5);
        assert_eq!(shape.ravel_index(&[0, 2, 3]), 11);
        assert_eq!(shape.ravel_index(&[1, 0, 0]), 12);
        assert_eq!(shape.ravel_index(&[1, 1, 1]), 17);
        assert_eq!(shape.ravel_index(&[1, 2, 3]), 23);

        let shape = TensorShape::new(vec![2, 2, 2, 2]);

        assert_eq!(shape.ravel_index(&[0, 0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 0, 1, 0]), 2);
        assert_eq!(shape.ravel_index(&[0, 0, 1, 1]), 3);
        assert_eq!(shape.ravel_index(&[0, 1, 0, 0]), 4);
        assert_eq!(shape.ravel_index(&[1, 0, 0, 0]), 8);
        assert_eq!(shape.ravel_index(&[1, 1, 1, 1]), 15);

        let shape = TensorShape::new(vec![10, 20, 30]);

        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 1, 0]), 30);
        assert_eq!(shape.ravel_index(&[1, 0, 0]), 600);
        assert_eq!(shape.ravel_index(&[5, 10, 15]), 5 * 600 + 10 * 30 + 15);
        assert_eq!(shape.ravel_index(&[9, 19, 29]), 9 * 600 + 19 * 30 + 29);

        let shape = TensorShape::new(vec![1, 1, 1]);

        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);

        let shape = TensorShape::new(vec![3, 4]);

        let expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

        let mut index = 0;
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(shape.ravel_index(&[i, j]), expected[index]);
                index += 1;
            }
        }
    }

    #[test]
    fn test_unravel_index() {
        let shape = TensorShape::new(vec![5]);

        assert_eq!(shape.unravel_index(0), vec![0]);
        assert_eq!(shape.unravel_index(1), vec![1]);
        assert_eq!(shape.unravel_index(2), vec![2]);
        assert_eq!(shape.unravel_index(3), vec![3]);
        assert_eq!(shape.unravel_index(4), vec![4]);

        let shape = TensorShape::new(vec![2, 3]);

        assert_eq!(shape.unravel_index(0), vec![0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 2]);
        assert_eq!(shape.unravel_index(3), vec![1, 0]);
        assert_eq!(shape.unravel_index(4), vec![1, 1]);
        assert_eq!(shape.unravel_index(5), vec![1, 2]);

        let shape = TensorShape::new(vec![2, 3, 4]);

        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 0, 2]);
        assert_eq!(shape.unravel_index(3), vec![0, 0, 3]);
        assert_eq!(shape.unravel_index(4), vec![0, 1, 0]);
        assert_eq!(shape.unravel_index(5), vec![0, 1, 1]);
        assert_eq!(shape.unravel_index(11), vec![0, 2, 3]);
        assert_eq!(shape.unravel_index(12), vec![1, 0, 0]);
        assert_eq!(shape.unravel_index(17), vec![1, 1, 1]);
        assert_eq!(shape.unravel_index(23), vec![1, 2, 3]);

        let shape = TensorShape::new(vec![2, 2, 2, 2]);

        assert_eq!(shape.unravel_index(0), vec![0, 0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 0, 1, 0]);
        assert_eq!(shape.unravel_index(3), vec![0, 0, 1, 1]);
        assert_eq!(shape.unravel_index(4), vec![0, 1, 0, 0]);
        assert_eq!(shape.unravel_index(8), vec![1, 0, 0, 0]);
        assert_eq!(shape.unravel_index(15), vec![1, 1, 1, 1]);

        let shape = TensorShape::new(vec![10, 20, 30]);

        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 1]);
        assert_eq!(shape.unravel_index(30), vec![0, 1, 0]);
        assert_eq!(shape.unravel_index(600), vec![1, 0, 0]);
        assert_eq!(shape.unravel_index(5 * 600 + 10 * 30 + 15), vec![5, 10, 15]);
        assert_eq!(shape.unravel_index(9 * 600 + 19 * 30 + 29), vec![9, 19, 29]);

        let shape = TensorShape::new(vec![1, 1, 1]);
        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);

        let shape = TensorShape::new(vec![]);
        assert_eq!(shape.unravel_index(0), vec![]);

        let shape = TensorShape::new(vec![3, 4]);

        let expected_indices = [
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
            vec![1, 3],
            vec![2, 0],
            vec![2, 1],
            vec![2, 2],
            vec![2, 3],
        ];

        for (flat_index, expected_multi_index) in expected_indices.iter().enumerate() {
            assert_eq!(shape.unravel_index(flat_index), *expected_multi_index);
        }

        let shape = TensorShape::new(vec![4, 5, 6]);

        for flat_index in 0..(4 * 5 * 6) {
            let multi_index = shape.unravel_index(flat_index);
            let recovered_flat_index = shape.ravel_index(&multi_index);
            assert_eq!(flat_index, recovered_flat_index);
        }
    }

    #[test]
    fn test_tensor_index() {
        let mut tensor_1d = Tensor::<f32>::zeros(vec![5]);
        assert_eq!(tensor_1d[&[0]], 0.0);
        assert_eq!(tensor_1d[&[4]], 0.0);

        tensor_1d[&[0]] = 1.0;
        tensor_1d[&[1]] = 2.0;
        tensor_1d[&[4]] = 5.0;

        assert_eq!(tensor_1d[&[0]], 1.0);
        assert_eq!(tensor_1d[&[1]], 2.0);
        assert_eq!(tensor_1d[&[2]], 0.0);
        assert_eq!(tensor_1d[&[4]], 5.0);

        let mut tensor_2d = Tensor::<i32>::zeros(vec![3, 4]);
        assert_eq!(tensor_2d[&[0, 0]], 0);
        assert_eq!(tensor_2d[&[2, 3]], 0);

        tensor_2d[&[0, 0]] = 10;
        tensor_2d[&[0, 3]] = 13;
        tensor_2d[&[1, 2]] = 42;
        tensor_2d[&[2, 3]] = 99;

        assert_eq!(tensor_2d[&[0, 0]], 10);
        assert_eq!(tensor_2d[&[0, 3]], 13);
        assert_eq!(tensor_2d[&[1, 2]], 42);
        assert_eq!(tensor_2d[&[2, 3]], 99);
        assert_eq!(tensor_2d[&[0, 1]], 0);
        assert_eq!(tensor_2d[&[1, 0]], 0);

        let mut tensor_3d = Tensor::<f64>::zeros(vec![2, 3, 4]);
        tensor_3d[&[0, 0, 0]] = 1.1;
        tensor_3d[&[1, 2, 3]] = 2.2;
        tensor_3d[&[0, 1, 2]] = 3.3;
        tensor_3d[&[1, 0, 1]] = 4.4;

        assert_eq!(tensor_3d[&[0, 0, 0]], 1.1);
        assert_eq!(tensor_3d[&[1, 2, 3]], 2.2);
        assert_eq!(tensor_3d[&[0, 1, 2]], 3.3);
        assert_eq!(tensor_3d[&[1, 0, 1]], 4.4);
        assert_eq!(tensor_3d[&[0, 0, 1]], 0.0);
        assert_eq!(tensor_3d[&[1, 1, 1]], 0.0);

        let mut tensor_mut = Tensor::<i32>::zeros(vec![2, 2]);
        {
            let value_ref = &mut tensor_mut[&[0, 1]];
            *value_ref = 42;
        }
        assert_eq!(tensor_mut[&[0, 1]], 42);

        tensor_mut[&[1, 0]] += 10;
        tensor_mut[&[1, 0]] *= 2;
        assert_eq!(tensor_mut[&[1, 0]], 20);
    }

    #[test]
    fn test_tensor_storage_and_consistency() {
        let mut storage = TensorStorage::<u8>::zeros(5);
        assert_eq!(storage[0], 0);
        assert_eq!(storage[4], 0);

        storage[0] = 100;
        storage[2] = 200;
        storage[4] = 255;

        assert_eq!(storage[0], 100);
        assert_eq!(storage[1], 0);
        assert_eq!(storage[2], 200);
        assert_eq!(storage[3], 0);
        assert_eq!(storage[4], 255);

        let mut tensor = Tensor::<i16>::zeros(vec![3, 4]);
        let test_cases = vec![
            ([0, 0], 100),
            ([0, 3], 103),
            ([1, 1], 111),
            ([2, 0], 200),
            ([2, 3], 203),
        ];

        for &(indices, value) in &test_cases {
            tensor[&indices] = value;
        }

        for &(indices, expected_value) in &test_cases {
            assert_eq!(
                tensor[&indices], expected_value,
                "Failed for indices {:?}",
                indices
            );
        }

        for &(indices, expected_value) in &test_cases {
            let flat_index = tensor.shape.ravel_index(&indices);
            assert_eq!(
                tensor.storage[flat_index], expected_value,
                "Flat index consistency failed for indices {:?}",
                indices
            );
        }

        let mut tensor_3d = Tensor::<f32>::zeros(vec![2, 3, 4]);
        let mut expected_value = 1.0;

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    tensor_3d[&[i, j, k]] = expected_value;
                    expected_value += 1.0;
                }
            }
        }

        expected_value = 1.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(
                        tensor_3d[&[i, j, k]],
                        expected_value,
                        "Failed at indices [{}, {}, {}]",
                        i,
                        j,
                        k
                    );
                    expected_value += 1.0;
                }
            }
        }
    }

    #[test]
    fn test_permute() {
        let shape_2d = TensorShape::new(vec![3, 4]);
        assert_eq!(shape_2d.shape, vec![3, 4]);
        assert_eq!(shape_2d.strides, vec![4, 1]);

        let shape_2d = shape_2d.permute(&[1, 0]);
        assert_eq!(shape_2d.shape, vec![4, 3]);
        assert_eq!(shape_2d.strides, vec![1, 4]);

        let shape_3d = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape_3d.shape, vec![2, 3, 4]);
        assert_eq!(shape_3d.strides, vec![12, 4, 1]);

        let shape_3d = shape_3d.permute(&[2, 0, 1]);
        assert_eq!(shape_3d.shape, vec![4, 2, 3]);
        assert_eq!(shape_3d.strides, vec![1, 12, 4]);

        let shape_4d = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape_4d.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape_4d.strides, vec![60, 20, 5, 1]);

        let shape_4d = shape_4d.permute(&[3, 2, 1, 0]);
        assert_eq!(shape_4d.shape, vec![5, 4, 3, 2]);
        assert_eq!(shape_4d.strides, vec![1, 5, 20, 60]);

        let shape_identity = TensorShape::new(vec![2, 3, 4]);
        let original_shape = shape_identity.shape.clone();
        let original_strides = shape_identity.strides.clone();

        let shape_identity = shape_identity.permute(&[0, 1, 2]);
        assert_eq!(shape_identity.shape, original_shape);
        assert_eq!(shape_identity.strides, original_strides);

        let shape_1d = TensorShape::new(vec![10]);
        assert_eq!(shape_1d.shape, vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let shape_1d = shape_1d.permute(&[0]);
        assert_eq!(shape_1d.shape, vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let original_shape = TensorShape::new(vec![2, 3, 4]);
        let permuted_shape = original_shape.permute(&[1, 2, 0]);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let original_flat = original_shape.ravel_index(&[i, j, k]);
                    let permuted_flat = permuted_shape.ravel_index(&[j, k, i]);
                    assert_eq!(
                        original_flat, permuted_flat,
                        "Index mismatch for [{}, {}, {}] vs [{}, {}, {}]",
                        i, j, k, j, k, i
                    );
                }
            }
        }

        let empty_shape = TensorShape::new(vec![]).permute(&[]);
        assert_eq!(empty_shape.shape, vec![]);
        assert_eq!(empty_shape.strides, vec![]);
    }

    #[test]
    fn test_merge() {
        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape.strides, vec![60, 20, 5, 1]);

        let merged_shape = shape.merge(1..=2);
        assert_eq!(merged_shape.shape, vec![2, 12, 5]);
        assert_eq!(merged_shape.strides, vec![60, 5, 1]);

        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        let merged_shape = shape.merge(1..=1);
        assert_eq!(merged_shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(merged_shape.strides, vec![60, 20, 5, 1]);

        let original_shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = original_shape.merge(1..=2);

        assert_eq!(
            original_shape.ravel_index(&[0, 0, 0]),
            merged_shape.ravel_index(&[0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[0, 1, 2]),
            merged_shape.ravel_index(&[0, 6])
        );
        assert_eq!(
            original_shape.ravel_index(&[1, 2, 3]),
            merged_shape.ravel_index(&[1, 11])
        );

        let shape = TensorShape::new(vec![10]);
        let merged_shape = shape.merge(0..=0);
        assert_eq!(merged_shape.shape, vec![10]);
        assert_eq!(merged_shape.strides, vec![1]);
    }

    #[test]
    fn test_split() {
        let shape = TensorShape::new(vec![2, 12, 5]);
        let split_shape = shape.split(1, &[3, 4]);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(split_shape.strides, vec![60, 20, 5, 1]);

        let shape = TensorShape::new(vec![24]);
        let split_shape = shape.split(0, &[2, 3, 0]);
        assert_eq!(split_shape.shape, vec![2, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 4, 1]);

        let shape = TensorShape::new(vec![2, 3, 24]);
        let split_shape = shape.split(2, &[4, 6]);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 6]);
        assert_eq!(split_shape.strides, vec![72, 24, 6, 1]);

        let shape = TensorShape::new(vec![12, 3, 4]);
        let split_shape = shape.split(0, &[3, 4]);
        assert_eq!(split_shape.shape, vec![3, 4, 3, 4]);
        assert_eq!(split_shape.strides, vec![48, 12, 4, 1]);

        let shape = TensorShape::new(vec![30]);
        let split_shape = shape.split(0, &[5, 6]);
        assert_eq!(split_shape.shape, vec![5, 6]);
        assert_eq!(split_shape.strides, vec![6, 1]);

        let shape = TensorShape::new(vec![2, 60, 3]);
        let split_shape = shape.split(1, &[4, 0, 5]);
        assert_eq!(split_shape.shape, vec![2, 4, 3, 5, 3]);
        assert_eq!(split_shape.strides, vec![180, 45, 15, 3, 1]);

        let original_shape = TensorShape::new(vec![6, 8]);
        let split_shape = original_shape.split(0, &[2, 3]).split(2, &[4, 2]);

        assert_eq!(
            original_shape.ravel_index(&[0, 0]),
            split_shape.ravel_index(&[0, 0, 0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[2, 6]),
            split_shape.ravel_index(&[0, 2, 3, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[5, 7]),
            split_shape.ravel_index(&[1, 2, 3, 1])
        );

        let shape = TensorShape::new(vec![4]);
        let split_shape = shape.split(0, &[4, 1]);
        assert_eq!(split_shape.shape, vec![4, 1]);
        assert_eq!(split_shape.strides, vec![1, 1]);

        let shape = TensorShape::new(vec![2, 3, 4]);
        let split_shape = shape.split(1, &[1, 3]);
        assert_eq!(split_shape.shape, vec![2, 1, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 12, 4, 1]);
    }

    #[test]
    fn test_slice() {
        let shape = TensorShape::new(vec![5, 6]);
        assert_eq!(shape.shape, vec![5, 6]);
        assert_eq!(shape.strides, vec![6, 1]);
        assert_eq!(shape.linear_offset, 0);

        let sliced_shape = shape.slice(0, 1..=3).slice(1, 2..=4);

        assert_eq!(sliced_shape.shape, vec![3, 3]);
        assert_eq!(sliced_shape.strides, vec![6, 1]);
        assert_eq!(sliced_shape.linear_offset, 6 + 2);

        let shape_3d = TensorShape::new(vec![4, 5, 6]);
        assert_eq!(shape_3d.strides, vec![30, 6, 1]);

        let sliced_3d = shape_3d.slice(0, 1..=2).slice(2, 1..=4);

        assert_eq!(sliced_3d.shape, vec![2, 5, 4]);
        assert_eq!(sliced_3d.strides, vec![30, 6, 1]);
        assert_eq!(sliced_3d.linear_offset, 30 + 1);

        let shape_1d = TensorShape::new(vec![10]);
        let sliced_1d = shape_1d.slice(0, 3..=7);

        assert_eq!(sliced_1d.shape, vec![5]);
        assert_eq!(sliced_1d.strides, vec![1]);
        assert_eq!(sliced_1d.linear_offset, 3);

        let shape_partial = TensorShape::new(vec![3, 4, 5]);
        let sliced_partial = shape_partial.slice(1, 1..=2);

        assert_eq!(sliced_partial.shape, vec![3, 2, 5]);
        assert_eq!(sliced_partial.strides, vec![20, 5, 1]);
        assert_eq!(sliced_partial.linear_offset, 5);

        let shape_single = TensorShape::new(vec![5, 5]);
        let sliced_single = shape_single.slice(0, 2..=2).slice(1, 3..=3);

        assert_eq!(sliced_single.shape, vec![1, 1]);
        assert_eq!(sliced_single.strides, vec![5, 1]);
        assert_eq!(sliced_single.linear_offset, 2 * 5 + 3);

        let original_shape = TensorShape::new(vec![4, 6]);
        let sliced_test = original_shape.slice(0, 1..=2).slice(1, 2..=4);

        let sliced_flat = sliced_test.ravel_index(&[0, 0]);
        let original_flat = original_shape.ravel_index(&[1, 2]);
        assert_eq!(sliced_flat, original_flat);

        let sliced_flat = sliced_test.ravel_index(&[1, 2]);
        let original_flat = original_shape.ravel_index(&[2, 4]);
        assert_eq!(sliced_flat, original_flat);
    }

    #[test]
    fn test_skip() {
        let shape_1d = TensorShape::new(vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let skipped_1d = shape_1d.skip(0, 2);
        assert_eq!(skipped_1d.shape, vec![5]);
        assert_eq!(skipped_1d.strides, vec![2]);
        assert_eq!(skipped_1d.linear_offset, 0);

        let shape_1d_odd = TensorShape::new(vec![9]);
        let skipped_1d_odd = shape_1d_odd.skip(0, 2);
        assert_eq!(skipped_1d_odd.shape, vec![5]);
        assert_eq!(skipped_1d_odd.strides, vec![2]);

        let shape_2d = TensorShape::new(vec![6, 8]);
        assert_eq!(shape_2d.strides, vec![8, 1]);

        let skipped_dim0 = shape_2d.skip(0, 2);
        assert_eq!(skipped_dim0.shape, vec![3, 8]);
        assert_eq!(skipped_dim0.strides, vec![16, 1]);
        assert_eq!(skipped_dim0.linear_offset, 0);

        let skipped_dim1 = shape_2d.skip(1, 3);
        assert_eq!(skipped_dim1.shape, vec![6, 3]);
        assert_eq!(skipped_dim1.strides, vec![8, 3]);
        assert_eq!(skipped_dim1.linear_offset, 0);

        let shape_3d = TensorShape::new(vec![4, 6, 8]);
        assert_eq!(shape_3d.strides, vec![48, 8, 1]);

        let skipped_3d = shape_3d.skip(1, 2);
        assert_eq!(skipped_3d.shape, vec![4, 3, 8]);
        assert_eq!(skipped_3d.strides, vec![48, 16, 1]);
        assert_eq!(skipped_3d.linear_offset, 0);

        let shape_chain = TensorShape::new(vec![8, 9]);
        let double_skipped = shape_chain.skip(0, 2).skip(1, 3);
        assert_eq!(double_skipped.shape, vec![4, 3]);
        assert_eq!(double_skipped.strides, vec![18, 3]);
        assert_eq!(double_skipped.linear_offset, 0);

        let shape_noop = TensorShape::new(vec![5, 7]);
        let no_change = shape_noop.skip(0, 1).skip(1, 1);
        assert_eq!(no_change.shape, shape_noop.shape);
        assert_eq!(no_change.strides, shape_noop.strides);
        assert_eq!(no_change.linear_offset, shape_noop.linear_offset);

        let test_cases = vec![(10, 2, 5), (10, 3, 4), (9, 3, 3), (7, 4, 2), (1, 2, 1)];

        for (original_size, step, expected_size) in test_cases {
            let shape = TensorShape::new(vec![original_size]);
            let skipped = shape.skip(0, step);
            assert_eq!(
                skipped.shape[0], expected_size,
                "Failed for {}.div_ceil({}) = {}",
                original_size, step, expected_size
            );
        }

        let shape_with_offset = TensorShape {
            shape: vec![6, 8],
            strides: vec![8, 1],
            linear_offset: 10,
        };

        let skipped_with_offset = shape_with_offset.skip(0, 3);
        assert_eq!(skipped_with_offset.shape, vec![2, 8]);
        assert_eq!(skipped_with_offset.strides, vec![24, 1]);
        assert_eq!(skipped_with_offset.linear_offset, 10);

        let original_shape = TensorShape::new(vec![6, 8]);
        let skipped_shape = original_shape.skip(1, 2);

        assert_eq!(skipped_shape.shape, vec![6, 4]);
        assert_eq!(skipped_shape.strides, vec![8, 2]);

        let skipped_flat = skipped_shape.ravel_index(&[1, 2]);
        let original_flat = original_shape.ravel_index(&[1, 4]);
        assert_eq!(skipped_flat, original_flat);

        let skipped_flat = skipped_shape.ravel_index(&[0, 1]);
        let original_flat = original_shape.ravel_index(&[0, 2]);
        assert_eq!(skipped_flat, original_flat);

        let skipped_flat = skipped_shape.ravel_index(&[2, 3]);
        let original_flat = original_shape.ravel_index(&[2, 6]);
        assert_eq!(skipped_flat, original_flat);
    }

    #[test]
    fn test_map() {
        let storage = TensorStorage {
            data: vec![1, 2, 3, 4, 5],
        };

        let mapped_storage = storage.map(|x| x * 2);
        assert_eq!(mapped_storage.data, vec![2, 4, 6, 8, 10]);

        let float_storage = storage.map(|x| *x as f32 + 0.5);
        assert_eq!(float_storage.data, vec![1.5, 2.5, 3.5, 4.5, 5.5]);

        let mut tensor = Tensor::<i32>::zeros(vec![2, 3]);
        tensor[&[0, 0]] = 1;
        tensor[&[0, 1]] = 2;
        tensor[&[0, 2]] = 3;
        tensor[&[1, 0]] = 4;
        tensor[&[1, 1]] = 5;
        tensor[&[1, 2]] = 6;

        let mapped_tensor = tensor.map(|x| x * x);

        assert_eq!(mapped_tensor.shape.shape, vec![2, 3]);
        assert_eq!(mapped_tensor.shape.strides, vec![3, 1]);

        assert_eq!(mapped_tensor[&[0, 0]], 1); // 1 * 1
        assert_eq!(mapped_tensor[&[0, 1]], 4); // 2 * 2
        assert_eq!(mapped_tensor[&[0, 2]], 9); // 3 * 3
        assert_eq!(mapped_tensor[&[1, 0]], 16); // 4 * 4
        assert_eq!(mapped_tensor[&[1, 1]], 25); // 5 * 5
        assert_eq!(mapped_tensor[&[1, 2]], 36); // 6 * 6

        let string_tensor = tensor.map(|x| format!("value_{}", x));
        assert_eq!(string_tensor[&[0, 0]], "value_1");
        assert_eq!(string_tensor[&[1, 2]], "value_6");

        let bool_tensor = tensor.map(|x| *x > 3);
        assert!(!bool_tensor[&[0, 0]]); // 1 > 3
        assert!(!bool_tensor[&[0, 1]]); // 2 > 3
        assert!(!bool_tensor[&[0, 2]]); // 3 > 3
        assert!(bool_tensor[&[1, 0]]); // 4 > 3
        assert!(bool_tensor[&[1, 1]]); // 5 > 3
        assert!(bool_tensor[&[1, 2]]); // 6 > 3
    }

    #[test]
    fn test_reduce() {
        // Test reducing a 1D tensor (should result in scalar)
        let mut tensor_1d = Tensor::<i32>::zeros(vec![5]);
        tensor_1d[&[0]] = 1;
        tensor_1d[&[1]] = 2;
        tensor_1d[&[2]] = 3;
        tensor_1d[&[3]] = 4;
        tensor_1d[&[4]] = 5;

        let sum_1d = tensor_1d.reduce(0, |a, b| a + b, false);
        assert_eq!(sum_1d.shape.shape, vec![]);
        assert!(sum_1d.shape.is_scalar());
        assert_eq!(sum_1d.storage.data, vec![15]);

        let max_1d = tensor_1d.reduce(0, |a, b| if a > b { *a } else { *b }, false);
        assert_eq!(max_1d.shape.shape, vec![]);
        assert_eq!(max_1d.storage.data, vec![5]);

        // Test reducing a 2D tensor along dimension 0 (rows)
        let mut tensor_2d = Tensor::<i32>::zeros(vec![3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                tensor_2d[&[i, j]] = (i * 4 + j + 1) as i32;
            }
        }

        let sum_rows = tensor_2d.reduce(0, |a, b| a + b, false);
        assert_eq!(sum_rows.shape.shape, vec![4]);
        assert_eq!(sum_rows[&[0]], 15);
        assert_eq!(sum_rows[&[1]], 18);
        assert_eq!(sum_rows[&[2]], 21);
        assert_eq!(sum_rows[&[3]], 24);

        // Test reducing the same 2D tensor along dimension 1 (columns)
        let sum_cols = tensor_2d.reduce(1, |a, b| a + b, false);
        assert_eq!(sum_cols.shape.shape, vec![3]);
        assert_eq!(sum_cols[&[0]], 10);
        assert_eq!(sum_cols[&[1]], 26);
        assert_eq!(sum_cols[&[2]], 42);

        // Test reducing a 3D tensor
        let mut tensor_3d = Tensor::<f32>::zeros(vec![2, 3, 4]);
        let mut value = 1.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    tensor_3d[&[i, j, k]] = value;
                    value += 1.0;
                }
            }
        }

        let reduced_dim0 = tensor_3d.reduce(0, |a, b| a + b, false);
        assert_eq!(reduced_dim0.shape.shape, vec![3, 4]);
        assert_eq!(reduced_dim0[&[0, 0]], 14.0);
        assert_eq!(reduced_dim0[&[0, 1]], 16.0);
        assert_eq!(reduced_dim0[&[2, 3]], 36.0);

        let reduced_dim1 = tensor_3d.reduce(1, |a, b| a + b, false);
        assert_eq!(reduced_dim1.shape.shape, vec![2, 4]);

        assert_eq!(reduced_dim1[&[0, 0]], 15.0);
        assert_eq!(reduced_dim1[&[0, 3]], 24.0);
        assert_eq!(reduced_dim1[&[1, 0]], 51.0);

        let reduced_dim2 = tensor_3d.reduce(2, |a, b| a + b, false);
        assert_eq!(reduced_dim2.shape.shape, vec![2, 3]);
        assert_eq!(reduced_dim2[&[0, 0]], 10.0);
        assert_eq!(reduced_dim2[&[0, 1]], 26.0);
        assert_eq!(reduced_dim2[&[1, 2]], 90.0);

        // Test with different reduction functions
        let mut small_tensor = Tensor::<i32>::zeros(vec![2, 3]);
        small_tensor[&[0, 0]] = 5;
        small_tensor[&[0, 1]] = 2;
        small_tensor[&[0, 2]] = 8;
        small_tensor[&[1, 0]] = 1;
        small_tensor[&[1, 1]] = 9;
        small_tensor[&[1, 2]] = 3;

        // Test max reduction
        let max_reduction = small_tensor.reduce(0, |a, b| if a > b { *a } else { *b }, false);
        assert_eq!(max_reduction.shape.shape, vec![3]);
        assert_eq!(max_reduction[&[0]], 5); // max(5, 1)
        assert_eq!(max_reduction[&[1]], 9); // max(2, 9)
        assert_eq!(max_reduction[&[2]], 8); // max(8, 3)

        // Test min reduction
        let min_reduction = small_tensor.reduce(1, |a, b| if a < b { *a } else { *b }, false);
        assert_eq!(min_reduction.shape.shape, vec![2]);
        assert_eq!(min_reduction[&[0]], 2); // min(5, 2, 8)
        assert_eq!(min_reduction[&[1]], 1); // min(1, 9, 3)

        // Test product reduction
        let product_reduction = small_tensor.reduce(0, |a, b| a * b, false);
        assert_eq!(product_reduction[&[0]], 5); // 5 * 1
        assert_eq!(product_reduction[&[1]], 18); // 2 * 9
        assert_eq!(product_reduction[&[2]], 24); // 8 * 3

        // Test edge case: single element tensor
        let mut single_tensor = Tensor::<i32>::zeros(vec![1]);
        single_tensor.storage.data[0] = 42;
        let single_reduced = single_tensor.reduce(0, |a, b| a + b, false);
        assert_eq!(single_reduced.shape.shape, vec![]);
        assert_eq!(single_reduced.storage.data, vec![42]);

        // Test edge case: tensor with dimension of size 1
        let mut narrow_tensor = Tensor::<i32>::zeros(vec![1, 5]);
        for j in 0..5 {
            narrow_tensor[&[0, j]] = j as i32 + 1;
        }

        let reduced_narrow = narrow_tensor.reduce(0, |a, b| a + b, false);
        assert_eq!(reduced_narrow.shape.shape, vec![5]);
        for j in 0..5 {
            assert_eq!(reduced_narrow[&[j]], (j as i32) + 1);
        }

        let reduced_narrow2 = narrow_tensor.reduce(1, |a, b| a + b, false);
        assert_eq!(reduced_narrow2.shape.shape, vec![1]);
        assert_eq!(reduced_narrow2[&[0]], 15); // 1+2+3+4+5

        // Test reduction on a permuted tensor
        let mut tensor_for_permute = Tensor::<i32>::zeros(vec![2, 3, 4]);
        let mut value = 1;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    tensor_for_permute[&[i, j, k]] = value;
                    value += 1;
                }
            }
        }

        let permuted_tensor = tensor_for_permute.permute(&[2, 0, 1]);
        assert_eq!(permuted_tensor.shape.shape, vec![4, 2, 3]);

        assert_eq!(permuted_tensor[&[0, 0, 0]], tensor_for_permute[&[0, 0, 0]]);
        assert_eq!(permuted_tensor[&[1, 0, 0]], tensor_for_permute[&[0, 0, 1]]);
        assert_eq!(permuted_tensor[&[0, 1, 0]], tensor_for_permute[&[1, 0, 0]]);

        let permuted_reduced = permuted_tensor.reduce(1, |a, b| a + b, false);
        assert_eq!(permuted_reduced.shape.shape, vec![4, 3]);

        assert_eq!(permuted_reduced[&[0, 0]], 1 + 13);
        assert_eq!(permuted_reduced[&[1, 0]], 2 + 14);
        assert_eq!(permuted_reduced[&[3, 2]], 12 + 24);

        // Test reduction on a sliced tensor
        let mut tensor_for_slice = Tensor::<i32>::zeros(vec![4, 5]);
        for i in 0..4 {
            for j in 0..5 {
                tensor_for_slice[&[i, j]] = (i * 5 + j + 1) as i32;
            }
        }

        let sliced_tensor = tensor_for_slice.slice(0, 1..=2).slice(1, 1..=3);
        assert_eq!(sliced_tensor.shape.shape, vec![2, 3]);

        assert_eq!(sliced_tensor[&[0, 0]], tensor_for_slice[&[1, 1]]);
        assert_eq!(sliced_tensor[&[1, 2]], tensor_for_slice[&[2, 3]]);

        let sliced_reduced = sliced_tensor.reduce(0, |a, b| a + b, false);
        assert_eq!(sliced_reduced.shape.shape, vec![3]);
        assert_eq!(
            sliced_reduced[&[0]],
            tensor_for_slice[&[1, 1]] + tensor_for_slice[&[2, 1]]
        );
        assert_eq!(
            sliced_reduced[&[1]],
            tensor_for_slice[&[1, 2]] + tensor_for_slice[&[2, 2]]
        );
        assert_eq!(
            sliced_reduced[&[2]],
            tensor_for_slice[&[1, 3]] + tensor_for_slice[&[2, 3]]
        );

        // Test reduction on a tensor with skip (non-contiguous strides)
        let mut tensor_for_skip = Tensor::<i32>::zeros(vec![6, 8]);
        for i in 0..6 {
            for j in 0..8 {
                tensor_for_skip[&[i, j]] = (i * 8 + j + 1) as i32;
            }
        }

        let skipped_tensor = tensor_for_skip.skip(0, 2).skip(1, 2);
        assert_eq!(skipped_tensor.shape.shape, vec![3, 4]);

        assert_eq!(skipped_tensor[&[0, 0]], tensor_for_skip[&[0, 0]]);
        assert_eq!(skipped_tensor[&[0, 1]], tensor_for_skip[&[0, 2]]);
        assert_eq!(skipped_tensor[&[1, 0]], tensor_for_skip[&[2, 0]]);
        assert_eq!(skipped_tensor[&[2, 3]], tensor_for_skip[&[4, 6]]);

        let skip_reduced = skipped_tensor.reduce(1, |a, b| a + b, false);
        assert_eq!(skip_reduced.shape.shape, vec![3]);
        assert_eq!(skip_reduced[&[0]], 1 + 3 + 5 + 7);
        assert_eq!(skip_reduced[&[1]], 17 + 19 + 21 + 23);
        assert_eq!(skip_reduced[&[2]], 33 + 35 + 37 + 39);

        // Test complex combination: permute + slice + reduce
        let mut complex_tensor = Tensor::<i32>::zeros(vec![3, 4, 5]);
        let mut val = 1;
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    complex_tensor[&[i, j, k]] = val;
                    val += 1;
                }
            }
        }

        let complex_permuted = complex_tensor.permute(&[2, 0, 1]);
        assert_eq!(complex_permuted.shape.shape, vec![5, 3, 4]);

        let complex_sliced = complex_permuted.slice(0, 1..=3);
        assert_eq!(complex_sliced.shape.shape, vec![3, 3, 4]);

        assert_eq!(complex_sliced[&[0, 0, 0]], complex_tensor[&[0, 0, 1]]);
        assert_eq!(complex_sliced[&[1, 1, 2]], complex_tensor[&[1, 2, 2]]);

        let complex_reduced = complex_sliced.reduce(2, |a, b| a + b, false);
        assert_eq!(complex_reduced.shape.shape, vec![3, 3]);

        let expected_00 = 2 + 7 + 12 + 17;
        assert_eq!(complex_reduced[&[0, 0]], expected_00);
    }

    #[test]
    fn test_reduce_keepdim() {
        let storage = TensorStorage {
            data: (0..6).collect::<Vec<i32>>(),
        };
        let shape = TensorShape::new(vec![2, 3]);
        let tensor = Tensor {
            shape: shape.clone(),
            storage: storage.clone(),
        };

        // Sum over dim=1 (the "3" dimension), without keepdim
        let reduced_no_keep = tensor.reduce(1, |a, b| a + b, false);
        println!("reduce(dim=1, keepdim=false):\n{}", reduced_no_keep);

        // Sum over dim=1, with keepdim
        let reduced_keep = tensor.reduce(1, |a, b| a + b, true);
        println!("reduce(dim=1, keepdim=true):\n{}", reduced_keep);

        // Expected:
        // tensor:
        // [[0, 1, 2],
        //  [3, 4, 5]]
        //
        // reduce(dim=1, keepdim=false):
        // [3, 12]           // shape [2]
        //
        // reduce(dim=1, keepdim=true):
        // [[3],
        //  [12]]            // shape [2,1]

        assert_eq!(reduced_no_keep.shape.shape, vec![2]);
        assert_eq!(reduced_keep.shape.shape, vec![2, 1]);

        assert_eq!(reduced_no_keep.storage.data, vec![3, 12]);
        assert_eq!(reduced_keep.storage.data, vec![3, 12]);
    }

    #[test]
    fn test_broadcast_op() {
        // Test basic element-wise addition with same shapes
        let mut tensor_a = Tensor::<i32>::zeros(vec![2, 3]);
        let mut tensor_b = Tensor::<i32>::zeros(vec![2, 3]);

        for i in 0..2 {
            for j in 0..3 {
                tensor_a[&[i, j]] = (i * 3 + j + 1) as i32;
            }
        }

        for i in 0..2 {
            for j in 0..3 {
                tensor_b[&[i, j]] = ((i * 3 + j + 1) * 10) as i32;
            }
        }

        let result = tensor_a.broadcast_op(&tensor_b, &[(0, 0), (1, 1)], |a, b| a + b);
        assert_eq!(result.shape.shape, vec![2, 3]);
        assert_eq!(result[&[0, 0]], 11);
        assert_eq!(result[&[0, 1]], 22);
        assert_eq!(result[&[1, 2]], 66);

        // Test broadcasting with dimension of size 1 - LHS preserving
        let mut tensor_2d = Tensor::<i32>::zeros(vec![2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                tensor_2d[&[i, j]] = (i * 3 + j + 1) as i32;
            }
        }

        let mut tensor_row = Tensor::<i32>::zeros(vec![1, 3]);
        tensor_row[&[0, 0]] = 100;
        tensor_row[&[0, 1]] = 200;
        tensor_row[&[0, 2]] = 300;

        let broadcast_result = tensor_2d.broadcast_op(&tensor_row, &[(0, 0), (1, 1)], |a, b| a + b);
        assert_eq!(broadcast_result.shape.shape, vec![2, 3]);
        assert_eq!(broadcast_result[&[0, 0]], 101);
        assert_eq!(broadcast_result[&[0, 1]], 202);
        assert_eq!(broadcast_result[&[0, 2]], 303);
        assert_eq!(broadcast_result[&[1, 0]], 104);
        assert_eq!(broadcast_result[&[1, 1]], 205);
        assert_eq!(broadcast_result[&[1, 2]], 306);

        // Test broadcasting in the other direction
        let mut tensor_col = Tensor::<i32>::zeros(vec![2, 1]);
        tensor_col[&[0, 0]] = 1000;
        tensor_col[&[1, 0]] = 2000;

        let col_broadcast_result =
            tensor_2d.broadcast_op(&tensor_col, &[(0, 0), (1, 1)], |a, b| a + b);
        assert_eq!(col_broadcast_result.shape.shape, vec![2, 3]);
        assert_eq!(col_broadcast_result[&[0, 0]], 1001);
        assert_eq!(col_broadcast_result[&[0, 1]], 1002);
        assert_eq!(col_broadcast_result[&[0, 2]], 1003);
        assert_eq!(col_broadcast_result[&[1, 0]], 2004);
        assert_eq!(col_broadcast_result[&[1, 1]], 2005);
        assert_eq!(col_broadcast_result[&[1, 2]], 2006);

        // Test with different operation (multiplication)
        let mult_result = tensor_2d.broadcast_op(&tensor_row, &[(0, 0), (1, 1)], |a, b| a * b);
        assert_eq!(mult_result.shape.shape, vec![2, 3]);
        assert_eq!(mult_result[&[0, 0]], 100);
        assert_eq!(mult_result[&[0, 1]], 400);
        assert_eq!(mult_result[&[0, 2]], 900);
        assert_eq!(mult_result[&[1, 0]], 400);
        assert_eq!(mult_result[&[1, 1]], 1000);
        assert_eq!(mult_result[&[1, 2]], 1800);

        // Test with tensors that have additional non-corresponding dimensions
        let mut tensor_3d = Tensor::<i32>::zeros(vec![2, 3, 4]);
        let mut tensor_1d = Tensor::<i32>::zeros(vec![3]);

        for i in 0..3 {
            tensor_1d[&[i]] = (i + 1) as i32;
        }

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    tensor_3d[&[i, j, k]] = (i * 12 + j * 4 + k + 1) as i32;
                }
            }
        }

        let mixed_result = tensor_3d.broadcast_op(&tensor_1d, &[(1, 0)], |a, b| a + b);
        assert_eq!(mixed_result.shape.shape, vec![2, 3, 4]);

        assert_eq!(mixed_result[&[0, 0, 0]], 2);
        assert_eq!(mixed_result[&[0, 1, 0]], 7);
        assert_eq!(mixed_result[&[0, 2, 0]], 12);
        assert_eq!(mixed_result[&[1, 0, 3]], 17);

        // Test LHS with non-corresponding dimensions + RHS dimensions
        let mut tensor_2x4 = Tensor::<i32>::zeros(vec![2, 4]);
        let mut tensor_3x5 = Tensor::<i32>::zeros(vec![3, 5]);

        for i in 0..2 {
            for j in 0..4 {
                tensor_2x4[&[i, j]] = (i * 4 + j + 1) as i32;
            }
        }

        for i in 0..3 {
            for j in 0..5 {
                tensor_3x5[&[i, j]] = ((i * 5 + j + 1) * 10) as i32;
            }
        }

        let mixed_dims_result = tensor_2x4.broadcast_op(&tensor_3x5, &[], |a, b| a + b);
        assert_eq!(mixed_dims_result.shape.shape, vec![2, 4, 3, 5]);

        let float_tensor = tensor_2d.map(|x| *x as f32);
        let int_to_float_result =
            tensor_2d.broadcast_op(&float_tensor, &[(0, 0), (1, 1)], |a, b| (*a as f32) + b);
        assert_eq!(int_to_float_result.shape.shape, vec![2, 3]);
        assert_eq!(int_to_float_result[&[0, 0]], 2.0);
        assert_eq!(int_to_float_result[&[1, 2]], 12.0);

        // Test broadcast_op with permuted tensors
        let mut base_tensor = Tensor::<i32>::zeros(vec![2, 3, 4]);
        let mut counter = 1;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    base_tensor[&[i, j, k]] = counter;
                    counter += 1;
                }
            }
        }

        let permuted_tensor = base_tensor.permute(&[2, 0, 1]);
        assert_eq!(permuted_tensor.shape.shape, vec![4, 2, 3]);

        let mut broadcast_1d = Tensor::<i32>::zeros(vec![2]);
        broadcast_1d[&[0]] = 100;
        broadcast_1d[&[1]] = 200;

        let permuted_broadcast =
            permuted_tensor.broadcast_op(&broadcast_1d, &[(1, 0)], |a, b| a + b);
        assert_eq!(permuted_broadcast.shape.shape, vec![4, 2, 3]);

        assert_eq!(
            permuted_broadcast[&[0, 0, 0]],
            base_tensor[&[0, 0, 0]] + 100
        );
        assert_eq!(
            permuted_broadcast[&[0, 1, 0]],
            base_tensor[&[1, 0, 0]] + 200
        );
        assert_eq!(
            permuted_broadcast[&[1, 0, 1]],
            base_tensor[&[0, 1, 1]] + 100
        );
        assert_eq!(
            permuted_broadcast[&[1, 1, 1]],
            base_tensor[&[1, 1, 1]] + 200
        );

        // Test broadcast_op with sliced tensors
        let mut slice_base = Tensor::<i32>::zeros(vec![4, 5]);
        for i in 0..4 {
            for j in 0..5 {
                slice_base[&[i, j]] = (i * 5 + j + 1) as i32;
            }
        }

        let sliced_tensor = slice_base.slice(0, 1..=2).slice(1, 1..=3);
        assert_eq!(sliced_tensor.shape.shape, vec![2, 3]);

        let mut slice_broadcast = Tensor::<i32>::zeros(vec![1, 3]);
        slice_broadcast[&[0, 0]] = 1000;
        slice_broadcast[&[0, 1]] = 2000;
        slice_broadcast[&[0, 2]] = 3000;

        let sliced_broadcast_result =
            sliced_tensor.broadcast_op(&slice_broadcast, &[(0, 0), (1, 1)], |a, b| a + b);
        assert_eq!(sliced_broadcast_result.shape.shape, vec![2, 3]); // LHS shape preserved

        assert_eq!(sliced_broadcast_result[&[0, 0]], slice_base[&[1, 1]] + 1000);
        assert_eq!(sliced_broadcast_result[&[0, 1]], slice_base[&[1, 2]] + 2000);
        assert_eq!(sliced_broadcast_result[&[0, 2]], slice_base[&[1, 3]] + 3000);
        assert_eq!(sliced_broadcast_result[&[1, 0]], slice_base[&[2, 1]] + 1000);
        assert_eq!(sliced_broadcast_result[&[1, 1]], slice_base[&[2, 2]] + 2000);
        assert_eq!(sliced_broadcast_result[&[1, 2]], slice_base[&[2, 3]] + 3000);

        // Test broadcast_op with skipped (strided) tensors
        let mut skip_base = Tensor::<i32>::zeros(vec![6, 8]);
        for i in 0..6 {
            for j in 0..8 {
                skip_base[&[i, j]] = (i * 8 + j + 1) as i32;
            }
        }

        let skipped_tensor = skip_base.skip(0, 2).skip(1, 2);
        assert_eq!(skipped_tensor.shape.shape, vec![3, 4]);

        let mut skip_col = Tensor::<i32>::zeros(vec![3, 1]);
        skip_col[&[0, 0]] = 10000;
        skip_col[&[1, 0]] = 20000;
        skip_col[&[2, 0]] = 30000;

        let skip_broadcast_result =
            skipped_tensor.broadcast_op(&skip_col, &[(0, 0), (1, 1)], |a, b| a * b);
        assert_eq!(skip_broadcast_result.shape.shape, vec![3, 4]);

        assert_eq!(skip_broadcast_result[&[0, 0]], skip_base[&[0, 0]] * 10000);
        assert_eq!(skip_broadcast_result[&[0, 1]], skip_base[&[0, 2]] * 10000);
        assert_eq!(skip_broadcast_result[&[1, 0]], skip_base[&[2, 0]] * 20000);
        assert_eq!(skip_broadcast_result[&[1, 3]], skip_base[&[2, 6]] * 20000);
        assert_eq!(skip_broadcast_result[&[2, 2]], skip_base[&[4, 4]] * 30000);

        // Test complex case: permute + slice + broadcast
        let mut complex_base = Tensor::<i32>::zeros(vec![3, 4, 5]);
        let mut val = 1;
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    complex_base[&[i, j, k]] = val;
                    val += 1;
                }
            }
        }

        let complex_permuted = complex_base.permute(&[2, 0, 1]);
        let complex_sliced = complex_permuted.slice(0, 1..=3);
        assert_eq!(complex_sliced.shape.shape, vec![3, 3, 4]);

        let mut complex_broadcast_tensor = Tensor::<i32>::zeros(vec![3]);
        complex_broadcast_tensor[&[0]] = 2;
        complex_broadcast_tensor[&[1]] = 3;
        complex_broadcast_tensor[&[2]] = 5;

        let complex_broadcast_result =
            complex_sliced.broadcast_op(&complex_broadcast_tensor, &[(1, 0)], |a, b| a * b);
        assert_eq!(complex_broadcast_result.shape.shape, vec![3, 3, 4]);

        assert_eq!(
            complex_broadcast_result[&[0, 0, 0]],
            complex_base[&[0, 0, 1]] * complex_broadcast_tensor[&[0]]
        );
        assert_eq!(
            complex_broadcast_result[&[1, 1, 2]],
            complex_base[&[1, 2, 2]] * complex_broadcast_tensor[&[1]]
        );
        assert_eq!(
            complex_broadcast_result[&[2, 2, 3]],
            complex_base[&[2, 3, 3]] * complex_broadcast_tensor[&[2]]
        );

        // Test broadcasting between two non-standard layout tensors
        let tensor_a_ns = base_tensor.permute(&[1, 2, 0]);
        let tensor_a_sliced = tensor_a_ns.slice(1, 1..=2);

        let tensor_b_ns = base_tensor.skip(0, 1).skip(2, 2);
        let tensor_b_permuted = tensor_b_ns.permute(&[1, 0, 2]);

        assert_eq!(tensor_a_sliced.shape.shape, vec![3, 2, 2]);
        assert_eq!(tensor_b_permuted.shape.shape, vec![3, 2, 2]);

        let ns_broadcast_result =
            tensor_a_sliced
                .broadcast_op(&tensor_b_permuted, &[(0, 0), (1, 1), (2, 2)], |a, b| a + b);
        assert_eq!(ns_broadcast_result.shape.shape, vec![3, 2, 2]);

        let a_val_000 = tensor_a_sliced[&[0, 0, 0]];
        let b_val_000 = tensor_b_permuted[&[0, 0, 0]];
        assert_eq!(ns_broadcast_result[&[0, 0, 0]], a_val_000 + b_val_000);

        let a_val_111 = tensor_a_sliced[&[1, 1, 1]];
        let b_val_111 = tensor_b_permuted[&[1, 1, 1]];
        assert_eq!(ns_broadcast_result[&[1, 1, 1]], a_val_111 + b_val_111);
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut matrix_a = Tensor::<i32>::zeros(vec![2, 3]);
        matrix_a[&[0, 0]] = 1;
        matrix_a[&[0, 1]] = 2;
        matrix_a[&[0, 2]] = 3;
        matrix_a[&[1, 0]] = 4;
        matrix_a[&[1, 1]] = 5;
        matrix_a[&[1, 2]] = 6;

        let mut matrix_b = Tensor::<i32>::zeros(vec![3, 2]);
        matrix_b[&[0, 0]] = 7;
        matrix_b[&[0, 1]] = 8;
        matrix_b[&[1, 0]] = 9;
        matrix_b[&[1, 1]] = 10;
        matrix_b[&[2, 0]] = 11;
        matrix_b[&[2, 1]] = 12;

        let intermediate = matrix_a.broadcast_op(&matrix_b, &[(1, 0)], |a, b| a * b);
        assert_eq!(intermediate.shape.shape, vec![2, 3, 2]);
        assert_eq!(intermediate[&[0, 0, 0]], 7);
        assert_eq!(intermediate[&[0, 1, 1]], 20);
        assert_eq!(intermediate[&[1, 2, 0]], 66);

        let result = intermediate.reduce(1, |a, b| a + b, false);
        assert_eq!(result.shape.shape, vec![2, 2]);
        assert_eq!(result[&[0, 0]], 58);
        assert_eq!(result[&[0, 1]], 64);
        assert_eq!(result[&[1, 0]], 139);
        assert_eq!(result[&[1, 1]], 154);

        let mut batch_a = Tensor::<i32>::zeros(vec![2, 3, 4]);
        let mut batch_b = Tensor::<i32>::zeros(vec![2, 4, 3]);

        for batch in 0..2 {
            for i in 0..3 {
                for j in 0..4 {
                    batch_a[&[batch, i, j]] = ((batch * 12) + (i * 4) + j + 1) as i32;
                }
            }
            for i in 0..4 {
                for j in 0..3 {
                    batch_b[&[batch, i, j]] = ((batch * 12) + (i * 3) + j + 13) as i32;
                }
            }
        }

        let batch_intermediate = batch_a.broadcast_op(&batch_b, &[(0, 0), (2, 1)], |a, b| a * b);
        assert_eq!(batch_intermediate.shape.shape, vec![2, 3, 4, 3]);

        let batch_result = batch_intermediate.reduce(2, |a, b| a + b, false);
        assert_eq!(batch_result.shape.shape, vec![2, 3, 3]);
        assert_eq!(
            batch_result[&[0, 0, 0]],
            batch_a[&[0, 0, 0]] * batch_b[&[0, 0, 0]]
                + batch_a[&[0, 0, 1]] * batch_b[&[0, 1, 0]]
                + batch_a[&[0, 0, 2]] * batch_b[&[0, 2, 0]]
                + batch_a[&[0, 0, 3]] * batch_b[&[0, 3, 0]]
        );
        assert_eq!(
            batch_result[&[0, 0, 1]],
            batch_a[&[0, 0, 0]] * batch_b[&[0, 0, 1]]
                + batch_a[&[0, 0, 1]] * batch_b[&[0, 1, 1]]
                + batch_a[&[0, 0, 2]] * batch_b[&[0, 2, 1]]
                + batch_a[&[0, 0, 3]] * batch_b[&[0, 3, 1]]
        );
        assert_eq!(
            batch_result[&[0, 0, 2]],
            batch_a[&[0, 0, 0]] * batch_b[&[0, 0, 2]]
                + batch_a[&[0, 0, 1]] * batch_b[&[0, 1, 2]]
                + batch_a[&[0, 0, 2]] * batch_b[&[0, 2, 2]]
                + batch_a[&[0, 0, 3]] * batch_b[&[0, 3, 2]]
        );
        assert_eq!(
            batch_result[&[1, 2, 2]],
            batch_a[&[1, 2, 0]] * batch_b[&[1, 0, 2]]
                + batch_a[&[1, 2, 1]] * batch_b[&[1, 1, 2]]
                + batch_a[&[1, 2, 2]] * batch_b[&[1, 2, 2]]
                + batch_a[&[1, 2, 3]] * batch_b[&[1, 3, 2]]
        );
    }

    #[test]
    fn test_softmax() {
        let mut tensor_2d = Tensor::<f32>::zeros(vec![2, 3]);
        tensor_2d[&[0, 0]] = 1.0;
        tensor_2d[&[0, 1]] = 2.0;
        tensor_2d[&[0, 2]] = 3.0;
        tensor_2d[&[1, 0]] = 4.0;
        tensor_2d[&[1, 1]] = 5.0;
        tensor_2d[&[1, 2]] = 6.0;

        let exp_tensor = tensor_2d.map(|x| x.exp());
        assert_eq!(exp_tensor.shape.shape, vec![2, 3]);
        assert_eq!(exp_tensor[&[0, 0]], 1_f32.exp());
        assert_eq!(exp_tensor[&[0, 2]], 3_f32.exp());

        let sum_tensor = exp_tensor.reduce(1, |a, b| a + b, false);
        assert_eq!(sum_tensor.shape.shape, vec![2]);
        assert_eq!(sum_tensor[&[0]], 1_f32.exp() + 2_f32.exp() + 3_f32.exp());
        assert_eq!(sum_tensor[&[1]], 4_f32.exp() + 5_f32.exp() + 6_f32.exp());

        let softmax_result = exp_tensor.broadcast_op(&sum_tensor, &[(0, 0)], |a, b| a / b);
        assert_eq!(softmax_result.shape.shape, vec![2, 3]);
        assert_eq!(softmax_result[&[0, 0]], 1_f32.exp() / sum_tensor[&[0]]);
        assert_eq!(softmax_result[&[0, 1]], 2_f32.exp() / sum_tensor[&[0]]);
        assert_eq!(softmax_result[&[0, 2]], 3_f32.exp() / sum_tensor[&[0]]);
        assert_eq!(softmax_result[&[1, 0]], 4_f32.exp() / sum_tensor[&[1]]);
        assert_eq!(softmax_result[&[1, 1]], 5_f32.exp() / sum_tensor[&[1]]);
        assert_eq!(softmax_result[&[1, 2]], 6_f32.exp() / sum_tensor[&[1]]);

        let mut tensor_3d = Tensor::<f32>::zeros(vec![2, 2, 4]);
        let mut value = 1.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..4 {
                    tensor_3d[&[i, j, k]] = value;
                    value += 1.0;
                }
            }
        }

        let exp_3d = tensor_3d.map(|x| x.exp());
        let sum_3d = exp_3d.reduce(2, |a, b| a + b, false);
        assert_eq!(sum_3d.shape.shape, vec![2, 2]);

        let softmax_3d = exp_3d.broadcast_op(&sum_3d, &[(0, 0), (1, 1)], |a, b| a / b);
        assert_eq!(softmax_3d.shape.shape, vec![2, 2, 4]);

        for i in 0..2 {
            for j in 0..2 {
                let row_sum: f32 = (0..4).map(|k| softmax_3d[&[i, j, k]]).sum();
                assert_eq!(row_sum, 1.0);

                for k in 0..4 {
                    let expected = exp_3d[&[i, j, k]] / sum_3d[&[i, j]];
                    assert_eq!(softmax_3d[&[i, j, k]], expected);
                }
            }
        }
    }

    #[test]
    fn test_one_hot_encode_basic() {
        let indices = Tensor::new(vec![0i32, 2, 1], vec![3]);
        let one_hot = Tensor::one_hot(&indices, 3);

        assert_eq!(one_hot.shape.shape, vec![3, 3]);
        assert_eq!(
            one_hot.storage.data,
            vec![
                1, 0, 0, // class 0
                0, 0, 1, // class 2
                0, 1, 0 // class 1
            ]
        );
    }

    #[test]
    fn test_one_hot_encode_2d() {
        let indices = Tensor::new(vec![0i32, 1, 2, 1], vec![2, 2]);
        let one_hot = Tensor::one_hot(&indices, 3);

        assert_eq!(one_hot.shape.shape, vec![2, 2, 3]);
        assert_eq!(
            one_hot.storage.data,
            [
                1, 0, 0, 0, 1, 0, // row 1
                0, 0, 1, 0, 1, 0 // row 2
            ]
        );
    }

    #[test]
    #[should_panic(expected = "num_classes must be > 0")]
    fn test_one_hot_invalid_num_classes() {
        let indices = Tensor::new(vec![0i32], vec![1]);
        Tensor::one_hot(&indices, 0);
    }

    #[test]
    #[should_panic(expected = "Index 3 out of bounds")]
    fn test_one_hot_out_of_bounds() {
        let indices = Tensor::new(vec![3i32], vec![1]);
        Tensor::one_hot(&indices, 3);
    }
}
