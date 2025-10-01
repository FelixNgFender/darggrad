use std::{
    fmt::{Debug, Display},
    ops::RangeInclusive,
};

#[derive(Debug, Clone)]
pub(crate) struct TensorShape {
    /// An n-tuple that represents the length of each dimension in the tensor.
    pub(crate) shape: Vec<usize>,
    /// Pre-computed strides for each dimension.
    pub(crate) strides: Vec<usize>,
    /// Pre-computed offset for linear indexing (useful when slicing).
    pub(crate) linear_offset: usize,
}

impl TensorShape {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        let mut strides = vec![1; shape.len()];
        (0..shape.len().saturating_sub(1)).rev().for_each(|i| {
            strides[i] = strides[i + 1] * shape[i + 1];
        });
        TensorShape {
            shape: shape.to_vec(),
            strides,
            linear_offset: 0,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn dim(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    fn is_vector(&self) -> bool {
        self.shape.len() == 1
    }

    /// Converts n-dimensional indices to a linear index.
    pub(crate) fn ravel_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Indices length must match tensor shape dimensions"
        );

        self.linear_offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>()
    }

    /// Converts a linear index to n-dimensional indices.
    pub(crate) fn unravel_index(&self, index: usize) -> Vec<usize> {
        if self.is_scalar() {
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
    pub(crate) fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            self.shape.len(),
            "Axes length must match tensor shape dimensions"
        );
        let shape = axes.iter().map(|&i| self.shape[i]).collect();
        let strides = axes.iter().map(|&i| self.strides[i]).collect();
        Self {
            shape,
            strides,
            linear_offset: self.linear_offset,
        }
    }

    /// Merges a range of dimensions into a single dimension.
    pub(crate) fn merge(&self, dim_range: RangeInclusive<usize>) -> Self {
        let (start, end) = (*dim_range.start(), *dim_range.end());

        assert!(
            start <= end && end < self.shape.len(),
            "Invalid dimension range for merging"
        );

        let merged_size = self.shape[start..=end].iter().product();
        let merged_stride = self.strides[end];

        let mut new_shape = Vec::with_capacity(self.shape.len() - (end - start));
        let mut new_strides = Vec::with_capacity(self.strides.len() - (end - start));

        new_shape.extend_from_slice(&self.shape[..start]);
        new_shape.push(merged_size);
        new_shape.extend_from_slice(&self.shape[end + 1..]);

        new_strides.extend_from_slice(&self.strides[..start]);
        new_strides.push(merged_stride);
        new_strides.extend_from_slice(&self.strides[end + 1..]);

        Self {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset,
        }
    }

    /// Splits a dimension into multiple dimensions according to the given shape.
    pub(crate) fn split(&self, dim: usize, shape: &[usize]) -> Self {
        assert!(dim < self.shape.len(), "Dimension index out of bounds");

        let original_size = self.shape[dim];
        let original_stride = self.strides[dim];

        // Calculate the product of non-zero sizes and find wildcard
        let mut non_zero_product = 1usize;
        let mut zero_index = None;

        for (i, &size) in shape.iter().enumerate() {
            if size == 0 {
                assert!(
                    zero_index.is_none(),
                    "Cannot have more than one wildcard (0) in split sizes"
                );
                zero_index = Some(i);
            } else {
                non_zero_product *= size;
            }
        }

        // Create the final sizes, inferring wildcards
        let mut final_sizes = shape.to_vec();
        if let Some(zero_index) = zero_index {
            assert!(
                original_size.is_multiple_of(non_zero_product),
                "Cannot split dimension of size {} into sizes {:?}",
                original_size,
                shape
            );
            let inferred_size = original_size / non_zero_product;
            final_sizes[zero_index] = inferred_size;
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        // Add dimensions before the split
        new_shape.extend_from_slice(&self.shape[..dim]);
        new_strides.extend_from_slice(&self.strides[..dim]);

        // Calculate strides for the split dimensions
        let mut current_stride = original_stride;
        for &size in final_sizes.iter().rev() {
            new_strides.push(current_stride);
            current_stride *= size;
        }

        // Reverse the strides we just added to maintain correct order
        let start_idx = new_strides.len() - final_sizes.len();
        new_strides[start_idx..].reverse();

        // Add the split dimensions to shape
        new_shape.extend_from_slice(&final_sizes);

        // Add remaining dimensions after the split
        if dim + 1 < self.shape.len() {
            new_shape.extend_from_slice(&self.shape[dim + 1..]);
            new_strides.extend_from_slice(&self.strides[dim + 1..]);
        }

        Self {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset,
        }
    }

    /// Slices the tensor along a specified dimension using an inclusive range.
    pub(crate) fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Self {
        assert!(dim < self.shape.len(), "Dimension index out of bounds");

        let start = *range.start();
        let end = *range.end();

        assert!(
            start <= end && end < self.shape[dim],
            "Invalid slice range for dimension {}",
            dim
        );

        let mut new_shape = self.shape.clone();
        new_shape[dim] = end - start + 1; // inclusive range

        let additional_offset = start * self.strides[dim];

        Self {
            shape: new_shape,
            strides: self.strides.clone(),
            linear_offset: self.linear_offset + additional_offset,
        }
    }

    /// Skips elements in a specified dimension by a given step size.
    pub(crate) fn skip(&self, dim: usize, step: usize) -> Self {
        // perform the equivalent of slicing with no range, but a step
        assert!(dim < self.shape.len(), "Dimension index out of bounds");

        let mut new_strides = self.strides.clone();
        new_strides[dim] *= step;

        let mut new_shape = self.shape.clone();
        new_shape[dim] = new_shape[dim].div_ceil(step);

        Self {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset,
        }
    }

    /// Broadcasts two tensor shapes together given corresponding dimensions.
    pub(crate) fn broadcast_shape(
        &self,
        other: &TensorShape,
        corresponding_dimensions: &[(usize, usize)],
    ) -> TensorShape {
        // Create mapping for corresponding dimensions
        let mut dim_correspondence = std::collections::HashMap::new();
        let mut other_dim_used = vec![false; other.shape.len()];

        for &(self_dim, other_dim) in corresponding_dimensions {
            assert!(
                self_dim < self.shape.len() && other_dim < other.shape.len(),
                "Dimension index out of bounds"
            );
            dim_correspondence.insert(self_dim, other_dim);
            other_dim_used[other_dim] = true;
        }

        // Build output shape: LHS dimensions (with broadcasting) + remaining RHS dimensions
        let mut output_shape = Vec::new();

        // Process LHS dimensions in order
        for (self_dim, &self_size) in self.shape.iter().enumerate() {
            if let Some(&other_dim) = dim_correspondence.get(&self_dim) {
                let other_size = other.shape[other_dim];

                if self_size == other_size {
                    output_shape.push(self_size);
                } else if self_size == 1 {
                    output_shape.push(other_size);
                } else if other_size == 1 {
                    output_shape.push(self_size);
                } else {
                    panic!(
                        "Cannot broadcast dimensions: {} and {}",
                        self_size, other_size
                    );
                }
            } else {
                output_shape.push(self_size);
            }
        }

        // Add remaining RHS dimensions that weren't used in correspondence
        for (other_dim, &other_size) in other.shape.iter().enumerate() {
            if !other_dim_used[other_dim] {
                output_shape.push(other_size);
            }
        }

        TensorShape::new(output_shape)
    }
}

impl From<&[usize]> for TensorShape {
    fn from(shape: &[usize]) -> Self {
        TensorShape::new(shape.to_vec())
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
