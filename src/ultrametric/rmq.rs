//! Range Maximum Query data structure
use std::{
    cmp::min,
    ops::{Bound, RangeBounds},
};

use num_traits::Float;

use crate::points::FloatType;

#[derive(Debug, Clone)]
/// Range Maximum Query data structure
/// with `O(1)` query time and `O(n log n)` space and construction time.
pub(super) struct Rmq<F: FloatType> {
    /// Dynamic programming table defined as `mem[l][i] = min(T[i..i + 2^l])`.
    mem: Vec<Vec<F>>,
}

impl<F: FloatType> Rmq<F> {
    /// Construct a new RMQ data structure over the given array.
    ///
    /// If `values` is empty, returns [`Err`].
    pub fn new(values: Vec<F>) -> Result<Self, ()> {
        if values.is_empty() {
            Err(())
        } else {
            let mem = build_rmq(values);
            Ok(Self { mem })
        }
    }

    /// Returns the maximum of the array in the specified range.
    ///
    /// Returns [`None`] if the range is empty.
    /// The endpoints of the range are clamped to the interval `[0..=n]`.
    pub fn get_max(&self, range: impl RangeBounds<usize>) -> Option<F> {
        let n = self.mem[0].len();
        let start = match range.start_bound() {
            Bound::Included(&idx) => min(n, idx),
            Bound::Excluded(&idx) => min(n, idx + 1),
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&idx) => min(n, idx + 1),
            Bound::Excluded(&idx) => min(n, idx),
            Bound::Unbounded => n,
        };
        if start >= end {
            return None;
        }
        let (i, j) = (start, end - 1);
        let delta = j + 1 - i;
        // `l` is the largest power of two less than or equal to `delta`.
        let l = (delta + 1).next_power_of_two() / 2;
        let step = l.ilog2() as usize;
        Some(Float::max(self.mem[step][i], self.mem[step][j + 1 - l]))
    }
}

/// Build the RMQ dynamic programming table.
///
/// The table is defined as `mem[step][i] = min(T[i..i + 2^step])`.
/// The algorithm uses the formula `mem[step+1][i] = min(mem[step][i], mem[step][i+2^step])`,
/// when `i+2^step < n`.
/// Takes `O(n log n)` time.
///
/// # Panics
/// Panics if `values` is empty.
fn build_rmq<F: FloatType>(values: Vec<F>) -> Vec<Vec<F>> {
    let n = values.len();
    let h = n.ilog2() as usize + 1;

    let mut mem: Vec<Vec<F>> = vec![values];

    for step in 0..(h - 1) {
        let l = 1 << step;
        let next = (0..n)
            .map(|i| {
                if i + l < n {
                    Float::max(mem[step][i], mem[step][i + l])
                } else {
                    mem[step][i]
                }
            })
            .collect();
        mem.push(next)
    }

    mem
}

#[cfg(test)]
mod test {
    use super::Rmq;

    use rand::{Rng, random};

    #[test]
    fn build_rmq() {
        let v = vec![1., 2., 3., 4., 5.];
        let _ = Rmq::new(v).unwrap();
    }

    #[test]
    fn test_random() {
        const N_ARRAYS: usize = 100;
        const ARR_LEN: usize = 1000;
        const N_ITER: usize = 50;
        for _ in 0..N_ARRAYS {
            let values: Vec<f32> = (0..ARR_LEN).map(|_| random()).collect();
            let rmq = Rmq::new(values.clone()).unwrap();
            for _ in 0..N_ITER {
                let i = rand::rng().random_range(0..ARR_LEN);
                let j = rand::rng().random_range((i + 1)..=ARR_LEN);
                let res = rmq.get_max(i..j).unwrap();
                assert_eq!(
                    values[i..j].iter().fold(f32::MIN, |a, b| f32::max(a, *b)),
                    res
                );
            }
        }
    }

    #[test]
    fn empty_range() {
        let v = vec![1., 2., 3., 4., 5.];
        let rmq = Rmq::new(v).unwrap();

        assert!(rmq.get_max(0..0).is_none());
        assert!(rmq.get_max(3..2).is_none());
        assert!(rmq.get_max(10..10).is_none());
    }

    #[test]
    fn extended_syntax() {
        let v = vec![1., 2., 3., 4., 5.];
        let rmq = Rmq::new(v).unwrap();

        assert!(rmq.get_max(..0).is_none());
        assert!(rmq.get_max(..=0).is_some());
        assert!(rmq.get_max(3..=3).is_some());
        assert!(rmq.get_max(4..).is_some());
        assert!(rmq.get_max(9..).is_none());
    }

    #[test]
    fn clamp_range() {
        let v = vec![1., 2., 3., 4., 5.];
        let rmq = Rmq::new(v).unwrap();

        assert!(rmq.get_max(1..1000).is_some());
    }
}
