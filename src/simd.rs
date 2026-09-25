//! 16-lane f32 vectors for the CPU evaluation engine. With AVX-512 enabled at
//! compile time (`target-cpu=native` on this machine) each value is one zmm
//! register and comparisons produce mask registers; elsewhere a portable
//! array fallback keeps the same interface.
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

pub const LANES: usize = 16;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod imp {
    use super::LANES;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct F(pub __m512);
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct M(pub __mmask16);

    impl F {
        #[inline(always)]
        pub fn splat(x: f32) -> F {
            unsafe { F(_mm512_set1_ps(x)) }
        }
        #[inline(always)]
        pub fn load(a: &[f32; LANES]) -> F {
            unsafe { F(_mm512_loadu_ps(a.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, a: &mut [f32; LANES]) {
            unsafe { _mm512_storeu_ps(a.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        pub fn sqrt(self) -> F {
            unsafe { F(_mm512_sqrt_ps(self.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: F) -> F {
            unsafe { F(_mm512_min_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: F) -> F {
            unsafe { F(_mm512_max_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn floor(self) -> F {
            unsafe {
                F(_mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(
                    self.0,
                ))
            }
        }
        #[inline(always)]
        pub fn abs(self) -> F {
            unsafe { F(_mm512_abs_ps(self.0)) }
        }
        /// self * a + b with one rounding.
        #[inline(always)]
        pub fn mul_add(self, a: F, b: F) -> F {
            unsafe { F(_mm512_fmadd_ps(self.0, a.0, b.0)) }
        }
        #[inline(always)]
        pub fn lt(self, o: F) -> M {
            unsafe { M(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn le(self, o: F) -> M {
            unsafe { M(_mm512_cmp_ps_mask::<_CMP_LE_OQ>(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn gt(self, o: F) -> M {
            unsafe { M(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn eq(self, o: F) -> M {
            unsafe { M(_mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, o.0)) }
        }
        /// Lanes where `m` is set take `a`, others `b`.
        #[inline(always)]
        pub fn select(m: M, a: F, b: F) -> F {
            unsafe { F(_mm512_mask_blend_ps(m.0, b.0, a.0)) }
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; LANES] {
            let mut a = [0.0; LANES];
            self.store(&mut a);
            a
        }
    }
    impl M {
        #[inline(always)]
        pub fn all_set() -> M {
            M(0xffff)
        }
        #[inline(always)]
        pub fn none() -> M {
            M(0)
        }
        #[inline(always)]
        pub fn any(self) -> bool {
            self.0 != 0
        }
    }
    macro_rules! binop {
        ($t:ident, $f:ident, $i:ident) => {
            impl std::ops::$t for F {
                type Output = F;
                #[inline(always)]
                fn $f(self, o: F) -> F {
                    unsafe { F($i(self.0, o.0)) }
                }
            }
        };
    }
    binop!(Add, add, _mm512_add_ps);
    binop!(Sub, sub, _mm512_sub_ps);
    binop!(Mul, mul, _mm512_mul_ps);
    binop!(Div, div, _mm512_div_ps);
    impl std::ops::BitAnd for M {
        type Output = M;
        #[inline(always)]
        fn bitand(self, o: M) -> M {
            M(self.0 & o.0)
        }
    }
    impl std::ops::BitOr for M {
        type Output = M;
        #[inline(always)]
        fn bitor(self, o: M) -> M {
            M(self.0 | o.0)
        }
    }
    impl std::ops::Not for M {
        type Output = M;
        #[inline(always)]
        fn not(self) -> M {
            M(!self.0)
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod imp {
    use super::LANES;

    #[derive(Clone, Copy)]
    pub struct F(pub [f32; LANES]);
    #[derive(Clone, Copy)]
    pub struct M(pub u16);

    #[inline(always)]
    fn map(a: F, f: impl Fn(f32) -> f32) -> F {
        F(std::array::from_fn(|i| f(a.0[i])))
    }
    #[inline(always)]
    fn zip(a: F, b: F, f: impl Fn(f32, f32) -> f32) -> F {
        F(std::array::from_fn(|i| f(a.0[i], b.0[i])))
    }
    #[inline(always)]
    fn cmp(a: F, b: F, f: impl Fn(f32, f32) -> bool) -> M {
        M((0..LANES).fold(0, |m, i| m | (u16::from(f(a.0[i], b.0[i])) << i)))
    }
    impl F {
        pub fn splat(x: f32) -> F {
            F([x; LANES])
        }
        pub fn load(a: &[f32; LANES]) -> F {
            F(*a)
        }
        pub fn store(self, a: &mut [f32; LANES]) {
            *a = self.0;
        }
        pub fn sqrt(self) -> F {
            map(self, f32::sqrt)
        }
        pub fn min(self, o: F) -> F {
            zip(self, o, f32::min)
        }
        pub fn max(self, o: F) -> F {
            zip(self, o, f32::max)
        }
        pub fn floor(self) -> F {
            map(self, f32::floor)
        }
        pub fn abs(self) -> F {
            map(self, f32::abs)
        }
        pub fn mul_add(self, a: F, b: F) -> F {
            F(std::array::from_fn(|i| self.0[i].mul_add(a.0[i], b.0[i])))
        }
        pub fn lt(self, o: F) -> M {
            cmp(self, o, |a, b| a < b)
        }
        pub fn le(self, o: F) -> M {
            cmp(self, o, |a, b| a <= b)
        }
        pub fn gt(self, o: F) -> M {
            cmp(self, o, |a, b| a > b)
        }
        pub fn eq(self, o: F) -> M {
            cmp(self, o, |a, b| a == b)
        }
        pub fn select(m: M, a: F, b: F) -> F {
            F(std::array::from_fn(|i| {
                if m.0 >> i & 1 == 1 { a.0[i] } else { b.0[i] }
            }))
        }
        pub fn to_array(self) -> [f32; LANES] {
            self.0
        }
    }
    impl M {
        pub fn all_set() -> M {
            M(0xffff)
        }
        pub fn none() -> M {
            M(0)
        }
        pub fn any(self) -> bool {
            self.0 != 0
        }
    }
    impl std::ops::Add for F {
        type Output = F;
        fn add(self, o: F) -> F {
            zip(self, o, |a, b| a + b)
        }
    }
    impl std::ops::Sub for F {
        type Output = F;
        fn sub(self, o: F) -> F {
            zip(self, o, |a, b| a - b)
        }
    }
    impl std::ops::Mul for F {
        type Output = F;
        fn mul(self, o: F) -> F {
            zip(self, o, |a, b| a * b)
        }
    }
    impl std::ops::Div for F {
        type Output = F;
        fn div(self, o: F) -> F {
            zip(self, o, |a, b| a / b)
        }
    }
    impl std::ops::BitAnd for M {
        type Output = M;
        fn bitand(self, o: M) -> M {
            M(self.0 & o.0)
        }
    }
    impl std::ops::BitOr for M {
        type Output = M;
        fn bitor(self, o: M) -> M {
            M(self.0 | o.0)
        }
    }
    impl std::ops::Not for M {
        type Output = M;
        fn not(self) -> M {
            M(!self.0)
        }
    }
}

pub use imp::{F, M};

impl Neg for F {
    type Output = F;
    #[inline(always)]
    fn neg(self) -> F {
        F::splat(0.0) - self
    }
}
impl AddAssign for F {
    #[inline(always)]
    fn add_assign(&mut self, o: F) {
        *self = *self + o;
    }
}
impl SubAssign for F {
    #[inline(always)]
    fn sub_assign(&mut self, o: F) {
        *self = *self - o;
    }
}
impl Add<f32> for F {
    type Output = F;
    #[inline(always)]
    fn add(self, o: f32) -> F {
        self + F::splat(o)
    }
}
impl Sub<f32> for F {
    type Output = F;
    #[inline(always)]
    fn sub(self, o: f32) -> F {
        self - F::splat(o)
    }
}
impl Mul<f32> for F {
    type Output = F;
    #[inline(always)]
    fn mul(self, o: f32) -> F {
        self * F::splat(o)
    }
}
impl Div<f32> for F {
    type Output = F;
    #[inline(always)]
    fn div(self, o: f32) -> F {
        self / F::splat(o)
    }
}
