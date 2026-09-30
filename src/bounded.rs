//! A vector with a fixed capacity that lives inline, without a heap
//! allocation. Creatures hold their genes in these (`evolution::Creature`),
//! so breeding a child never calls the allocator, and a clone copies only
//! the elements in use.
//!
//! Elements are `Copy`, so nothing ever needs dropping. Going past the
//! capacity is a bug, as indexing past the end of a slice is: `push` and the
//! other growing methods panic. `try_push` reports it instead.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

pub struct Bounded<T: Copy, const N: usize> {
    len: u32,
    items: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> Bounded<T, N> {
    pub const CAPACITY: usize = N;

    pub const fn new() -> Self {
        Self {
            len: 0,
            items: [MaybeUninit::uninit(); N],
        }
    }
    /// A copy of `items`. Panics if there are more than `N`.
    pub fn from_slice(items: &[T]) -> Self {
        let mut out = Self::new();
        out.extend_from_slice(items);
        out
    }
    pub const fn capacity(&self) -> usize {
        N
    }
    pub fn is_full(&self) -> bool {
        self.len as usize == N
    }
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the first `len` items are initialized.
        unsafe { std::slice::from_raw_parts(self.items.as_ptr().cast::<T>(), self.len as usize) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the first `len` items are initialized.
        unsafe {
            std::slice::from_raw_parts_mut(self.items.as_mut_ptr().cast::<T>(), self.len as usize)
        }
    }
    /// Appends `value`, or returns false when the array is full.
    #[inline]
    pub fn try_push(&mut self, value: T) -> bool {
        let len = self.len as usize;
        if len == N {
            return false;
        }
        self.items[len] = MaybeUninit::new(value);
        self.len += 1;
        true
    }
    #[inline]
    pub fn push(&mut self, value: T) {
        assert!(self.try_push(value), "Bounded<_, {N}> is full");
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: the item was initialized and is now past the end.
        Some(unsafe { self.items[self.len as usize].assume_init() })
    }
    pub fn insert(&mut self, index: usize, value: T) {
        let len = self.len as usize;
        assert!(index <= len, "insert index {index} past length {len}");
        assert!(len < N, "Bounded<_, {N}> is full");
        self.items.copy_within(index..len, index + 1);
        self.items[index] = MaybeUninit::new(value);
        self.len += 1;
    }
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len as usize;
        assert!(index < len, "remove index {index} past length {len}");
        let value = self[index];
        self.items.copy_within(index + 1..len, index);
        self.len -= 1;
        value
    }
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len as usize;
        assert!(index < len, "swap_remove index {index} past length {len}");
        let value = self[index];
        self.items[index] = self.items[len - 1];
        self.len -= 1;
        value
    }
    pub fn truncate(&mut self, len: usize) {
        if len < self.len as usize {
            self.len = len as u32;
        }
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    /// Keeps the items for which `keep` is true, in order.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.retain_mut(|item| keep(item));
    }
    /// Keeps the items for which `keep` is true, in order; `keep` may change
    /// the items it sees.
    pub fn retain_mut(&mut self, mut keep: impl FnMut(&mut T) -> bool) {
        let len = self.len as usize;
        let mut kept = 0;
        for index in 0..len {
            // SAFETY: items below `len` are initialized.
            let mut item = unsafe { self.items[index].assume_init() };
            if keep(&mut item) {
                self.items[kept] = MaybeUninit::new(item);
                kept += 1;
            }
        }
        self.len = kept as u32;
    }
    pub fn extend_from_slice(&mut self, items: &[T]) {
        let len = self.len as usize;
        assert!(
            len + items.len() <= N,
            "Bounded<_, {N}> is full: {len} + {}",
            items.len()
        );
        // SAFETY: MaybeUninit<T> has the layout of T.
        let source = unsafe {
            std::slice::from_raw_parts(items.as_ptr().cast::<MaybeUninit<T>>(), items.len())
        };
        self.items[len..len + items.len()].copy_from_slice(source);
        self.len += items.len() as u32;
    }
    /// Sets the length to `len`, filling new places with `value`.
    pub fn resize(&mut self, len: usize, value: T) {
        assert!(len <= N, "Bounded<_, {N}> cannot hold {len}");
        for index in self.len as usize..len {
            self.items[index] = MaybeUninit::new(value);
        }
        self.len = len as u32;
    }
    /// `len` copies of `value`.
    pub fn filled(len: usize, value: T) -> Self {
        let mut out = Self::new();
        out.resize(len, value);
        out
    }
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<T: Copy, const N: usize> Default for Bounded<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> Clone for Bounded<T, N> {
    fn clone(&self) -> Self {
        let mut out = Self::new();
        let len = self.len as usize;
        out.items[..len].copy_from_slice(&self.items[..len]);
        out.len = self.len;
        out
    }
    fn clone_from(&mut self, source: &Self) {
        let len = source.len as usize;
        self.items[..len].copy_from_slice(&source.items[..len]);
        self.len = source.len;
    }
}

impl<T: Copy, const N: usize> Deref for Bounded<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: Copy, const N: usize> DerefMut for Bounded<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
impl<T: Copy, const N: usize> AsRef<[T]> for Bounded<T, N> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq for Bounded<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Copy + PartialEq, const N: usize> PartialEq<[T]> for Bounded<T, N> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}
impl<T: Copy + PartialEq, const N: usize> PartialEq<Vec<T>> for Bounded<T, N> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Copy + Eq, const N: usize> Eq for Bounded<T, N> {}
impl<T: Copy + std::hash::Hash, const N: usize> std::hash::Hash for Bounded<T, N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: Copy + std::fmt::Debug, const N: usize> std::fmt::Debug for Bounded<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Copy, const N: usize> Extend<T> for Bounded<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}
impl<'a, T: Copy + 'a, const N: usize> Extend<&'a T> for Bounded<T, N> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for item in iter {
            self.push(*item);
        }
    }
}
impl<T: Copy, const N: usize> FromIterator<T> for Bounded<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::new();
        out.extend(iter);
        out
    }
}
impl<T: Copy, const N: usize> From<&[T]> for Bounded<T, N> {
    fn from(items: &[T]) -> Self {
        Self::from_slice(items)
    }
}
impl<T: Copy, const N: usize> From<Vec<T>> for Bounded<T, N> {
    fn from(items: Vec<T>) -> Self {
        Self::from_slice(&items)
    }
}
impl<T: Copy, const N: usize, const M: usize> From<[T; M]> for Bounded<T, N> {
    fn from(items: [T; M]) -> Self {
        Self::from_slice(&items)
    }
}

/// Iterator over a `Bounded` taken by value.
pub struct IntoIter<T: Copy, const N: usize> {
    items: Bounded<T, N>,
    next: usize,
}
impl<T: Copy, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let item = self.items.get(self.next).copied();
        self.next += 1;
        item
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.items.len().saturating_sub(self.next);
        (left, Some(left))
    }
}
impl<T: Copy, const N: usize> ExactSizeIterator for IntoIter<T, N> {}
impl<T: Copy, const N: usize> IntoIterator for Bounded<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;
    fn into_iter(self) -> IntoIter<T, N> {
        IntoIter {
            items: self,
            next: 0,
        }
    }
}
impl<'a, T: Copy, const N: usize> IntoIterator for &'a Bounded<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}
impl<'a, T: Copy, const N: usize> IntoIterator for &'a mut Bounded<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// Saved as a sequence, the same bytes a `Vec` writes.
impl<T: Copy + Serialize, const N: usize> Serialize for Bounded<T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(serializer)
    }
}
impl<'de, T: Copy + Deserialize<'de>, const N: usize> Deserialize<'de> for Bounded<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visit<T, const N: usize>(std::marker::PhantomData<T>);
        impl<'de, T: Copy + Deserialize<'de>, const N: usize> serde::de::Visitor<'de> for Visit<T, N> {
            type Value = Bounded<T, N>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a sequence of at most {N} items")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = Bounded::new();
                while let Some(item) = seq.next_element()? {
                    if !out.try_push(item) {
                        return Err(serde::de::Error::invalid_length(N + 1, &self));
                    }
                }
                Ok(out)
            }
        }
        deserializer.deserialize_seq(Visit::<T, N>(std::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behaves_like_a_vec() {
        let mut b: Bounded<u32, 8> = Bounded::new();
        let mut v: Vec<u32> = Vec::new();
        for i in 0..6 {
            b.push(i * 3);
            v.push(i * 3);
        }
        assert_eq!(b, v);
        b.insert(2, 99);
        v.insert(2, 99);
        assert_eq!(b.remove(4), v.remove(4));
        assert_eq!(b.swap_remove(1), v.swap_remove(1));
        b.retain(|x| x % 2 == 1);
        v.retain(|x| x % 2 == 1);
        assert_eq!(b, v);
        b.extend_from_slice(&[7, 8]);
        v.extend_from_slice(&[7, 8]);
        b.retain_mut(|x| {
            *x += 1;
            *x != 9
        });
        v.retain_mut(|x| {
            *x += 1;
            *x != 9
        });
        assert_eq!(b, v);
        assert_eq!(b.pop(), v.pop());
        b.truncate(1);
        v.truncate(1);
        assert_eq!(b, v);
        let c = b.clone();
        assert_eq!(c, b);
        assert_eq!(b.into_iter().collect::<Vec<_>>(), v);
    }

    #[test]
    fn full_arrays_refuse_more() {
        let mut b: Bounded<u8, 2> = [1u8, 2].into();
        assert!(b.is_full());
        assert!(!b.try_push(3));
        assert_eq!(b.len(), 2);
        let pushed = std::panic::catch_unwind(move || b.push(3));
        assert!(pushed.is_err());
    }

    #[test]
    fn saves_the_same_bytes_as_a_vec() {
        let v: Vec<u16> = vec![4, 5, 6];
        let b: Bounded<u16, 4> = Bounded::from_slice(&v);
        assert_eq!(
            bincode::serialize(&v).unwrap(),
            bincode::serialize(&b).unwrap()
        );
        let back: Bounded<u16, 4> = bincode::deserialize(&bincode::serialize(&v).unwrap()).unwrap();
        assert_eq!(back, b);
        let long: Vec<u16> = vec![1; 5];
        assert!(
            bincode::deserialize::<Bounded<u16, 4>>(&bincode::serialize(&long).unwrap()).is_err()
        );
    }
}
