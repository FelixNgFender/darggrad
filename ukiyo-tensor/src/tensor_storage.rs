use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

use num_traits::Zero;

use crate::tensor_shape::TensorShape;

#[derive(Debug, Clone)]
pub(crate) struct TensorStorage<T> {
    pub(crate) data: Vec<T>,
}

impl<T> TensorStorage<T> {
    pub(crate) fn map<F, U>(&self, f: F) -> TensorStorage<U>
    where
        F: Fn(&T) -> U,
    {
        TensorStorage {
            data: self.data.iter().map(f).collect(),
        }
    }
}

impl<T: Clone> TensorStorage<T> {
    pub(crate) fn reduce<F>(
        &self,
        shape: &TensorShape,
        dim: usize,
        f: F,
        keepdim: bool,
    ) -> TensorStorage<T>
    where
        F: Fn(&T, &T) -> T,
        T: Zero,
    {
        // Calculate result shape by removing the reduction dimension
        let mut result_shape_vec = shape.shape.clone();
        if keepdim || result_shape_vec.len() == 1 {
            result_shape_vec[dim] = 1;
        } else {
            result_shape_vec.remove(dim);
        }

        if result_shape_vec.is_empty() {
            // Reducing to scalar
            let v = self
                .data
                .iter()
                .cloned()
                .reduce(|a, b| f(&a, &b))
                .expect("Cannot reduce an empty tensor storage");
            return TensorStorage { data: vec![v] };
        }

        let result_shape = TensorShape::new(result_shape_vec);
        let mut result_storage = TensorStorage::zeros(result_shape.size());

        // Iterate through all positions in the result tensor
        for result_flat_idx in 0..result_shape.size() {
            let result_multi_idx = result_shape.unravel_index(result_flat_idx);

            // For each result position, reduce along the specified dimension
            let mut accumulated_value: Option<T> = None;

            for dim_idx in 0..shape.shape[dim] {
                // Reconstruct the full multi-index by inserting the dimension index
                let mut full_multi_idx = Vec::with_capacity(shape.shape.len());
                let mut result_idx_pos = 0;

                for d in 0..shape.shape.len() {
                    if d == dim {
                        full_multi_idx.push(dim_idx);
                    } else {
                        full_multi_idx.push(result_multi_idx[result_idx_pos]);
                        result_idx_pos += 1;
                    }
                }

                let source_flat_idx = shape.ravel_index(&full_multi_idx);
                let value = &self[source_flat_idx];

                accumulated_value = match accumulated_value {
                    None => Some(value.clone()),
                    Some(acc) => Some(f(&acc, value)),
                };
            }

            result_storage[result_flat_idx] = accumulated_value.unwrap();
        }

        result_storage
    }

    /// Performs an element-wise binary operation between two tensors with broadcasting.
    pub(crate) fn broadcast_op<F, U, T2>(
        &self,
        self_shape: &TensorShape,
        other: &TensorStorage<T2>,
        other_shape: &TensorShape,
        corresponding_dimensions: &[(usize, usize)],
        f: F,
    ) -> TensorStorage<U>
    where
        F: Fn(&T, &T2) -> U,
        U: Zero + Clone,
    {
        let result_shape = self_shape.broadcast_shape(other_shape, corresponding_dimensions);
        let mut result = TensorStorage::<U>::zeros(result_shape.size());

        // Create mapping for corresponding dimensions
        let mut dim_correspondence = std::collections::HashMap::new();
        let mut other_dim_used = vec![false; other_shape.shape.len()];

        for &(self_dim, other_dim) in corresponding_dimensions {
            dim_correspondence.insert(self_dim, other_dim);
            other_dim_used[other_dim] = true;
        }

        // Perform the element-wise operation
        for flat_idx in 0..result_shape.size() {
            let output_multi_idx = result_shape.unravel_index(flat_idx);

            // Map output indices to input tensor indices
            let mut self_idx = vec![0; self_shape.shape.len()];
            let mut other_idx = vec![0; other_shape.shape.len()];

            // Map LHS dimensions
            for (self_dim, &self_size) in self_shape.shape.iter().enumerate() {
                let output_val = output_multi_idx[self_dim];

                if let Some(&other_dim) = dim_correspondence.get(&self_dim) {
                    if self_size == 1 {
                        self_idx[self_dim] = 0;
                    } else {
                        self_idx[self_dim] = output_val;
                    }

                    if other_shape.shape[other_dim] == 1 {
                        other_idx[other_dim] = 0;
                    } else {
                        other_idx[other_dim] = output_val;
                    }
                } else {
                    self_idx[self_dim] = output_val;
                }
            }

            // Map remaining RHS dimensions
            let mut rhs_output_offset = self_shape.shape.len();
            for (other_dim, _) in other_shape.shape.iter().enumerate() {
                if !other_dim_used[other_dim] {
                    other_idx[other_dim] = output_multi_idx[rhs_output_offset];
                    rhs_output_offset += 1;
                }
            }

            // Get values from input tensors using proper indexing
            let self_flat = self_shape.ravel_index(&self_idx);
            let other_flat = other_shape.ravel_index(&other_idx);

            let self_val = &self[self_flat];
            let other_val = &other[other_flat];

            // Apply operation and store result
            result[flat_idx] = f(self_val, other_val);
        }

        result
    }
}

impl<T: Zero + Clone> TensorStorage<T> {
    pub(crate) fn zeros(size: usize) -> Self {
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
