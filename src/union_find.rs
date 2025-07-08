use std::mem::swap;

pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next: Vec<Option<usize>>,
    last: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        let size = vec![1; n];
        let next = vec![None; n];
        let last = (0..n).collect();

        Self {
            parent,
            size,
            next,
            last,
        }
    }

    pub fn find(&self, x: usize) -> usize {
        let mut r = x;
        // find root
        while self.parent[r] != r {
            r = self.parent[r];
        }

        r
    }

    pub fn merge(&mut self, mut i: usize, mut j: usize) -> Option<usize> {
        // Balanced binary tree algorithm: make the root of the smallest cluster
        // point to the root of the largest cluster.
        i = self.find(i);
        j = self.find(j);

        if i == j {
            return None;
        }

        if self.size[i] < self.size[j] {
            swap(&mut i, &mut j);
        }
        // i now has the largest cluster: connect j to i
        self.parent[j] = i;
        self.size[i] += self.size[j];
        self.next[self.last[i]] = Some(j);
        self.last[i] = self.last[j];

        Some(i)
    }

    /// Merges two clusters and returns the last element in the resulting cluster,
    /// for the iterating order over clusters (e.g. iteration with [`Self::iter_cluster`]).
    fn merge_ret_last(&mut self, mut i: usize, mut j: usize) -> Option<usize> {
        i = self.find(i);
        j = self.find(j);

        if i == j {
            return None;
        }

        if self.size[i] < self.size[j] {
            swap(&mut i, &mut j);
        }
        // i now has the largest cluster: connect j to i
        self.parent[j] = i;
        self.size[i] += self.size[j];
        let tmp = self.last[i];
        self.last[i] = self.last[j];
        self.next[tmp] = Some(j);

        Some(tmp)
    }

    pub fn cluster_size(&self, i: usize) -> usize {
        self.size[self.find(i)]
    }

    pub fn iter_cluster(&self, i: usize) -> ClutserIter {
        ClutserIter {
            uf: self,
            next: Some(self.find(i)),
        }
    }
}

pub struct ClutserIter<'u> {
    uf: &'u UnionFind,
    next: Option<usize>,
}

impl<'u> Iterator for ClutserIter<'u> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.next {
            self.next = self.uf.next[i];
            Some(i)
        } else {
            None
        }
    }
}

/// Union-find data structure, with data between elements.
pub struct UnionFindWithData<T> {
    uf: UnionFind,
    data: Vec<Option<T>>,
}

impl<T> UnionFindWithData<T> {
    pub fn new(n: usize) -> Self {
        let uf = UnionFind::new(n);
        let data = (0..n).map(|_| None).collect();

        Self { uf, data }
    }

    pub fn merge(&mut self, i: usize, j: usize, data: T) -> Option<usize> {
        self.uf
            .merge_ret_last(i, j)
            .inspect(|last| self.data[*last] = Some(data))
    }

    pub fn iter_data(&self, i: usize) -> ClutserDataIter<'_, T> {
        ClutserDataIter {
            uf: self,
            next: Some(self.uf.find(i)),
        }
    }

    pub fn iter_cluster(&self, i: usize) -> ClutserIter {
        self.uf.iter_cluster(i)
    }
}

pub struct ClutserDataIter<'u, T> {
    uf: &'u UnionFindWithData<T>,
    next: Option<usize>,
}

impl<'u, T: Copy> Iterator for ClutserDataIter<'u, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.next {
            self.next = self.uf.uf.next[i];
            self.uf.data[i]
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_iteration() {
        let mut uf = UnionFind::new(10);
        uf.merge(1, 5);
        uf.merge(2, 3);
        uf.merge(1, 7);
        uf.merge(5, 9);

        let mut cluster = uf.iter_cluster(5).collect::<Vec<_>>();
        cluster.sort();
        assert_eq!(cluster, &[1, 5, 7, 9]);

        assert_eq!(uf.cluster_size(5), 4);
        assert_eq!(uf.cluster_size(2), 2);
        assert_eq!(uf.cluster_size(0), 1);
    }

    #[test]
    fn data_iteration() {
        let mut uf = UnionFindWithData::new(10);
        uf.merge(1, 2, 10);
        uf.merge(5, 7, 8);
        uf.merge(2, 3, 5);
        uf.merge(1, 4, 1);

        let cluster_data = uf.iter_data(1).collect::<Vec<_>>();
        assert_eq!(cluster_data, &[10, 5, 1]);
    }
}
