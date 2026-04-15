//! Optimized f32 <-> i16 conversions for Opus codec

const SCALE: f32 = 32768.0;
const INV_SCALE: f32 = 1.0 / 32768.0;
const I16_MIN_F: f32 = -32768.0;
const I16_MAX_F: f32 = 32767.0;

#[inline(always)]
pub fn f32_to_i16(x: f32) -> i16 {
    (x * SCALE).clamp(I16_MIN_F, I16_MAX_F) as i16
}

#[inline(always)]
pub fn i16_to_f32(x: i16) -> f32 {
    x as f32 * INV_SCALE
}

#[cfg(target_arch = "aarch64")]
mod simd {
    use super::*;
    use std::arch::aarch64::*;

    pub fn convert_f32_to_i16_neon(src: &[f32], dst: &mut [i16]) {
        let len = src.len();
        let mut i = 0;
        unsafe {
            let scale = vdupq_n_f32(SCALE);
            let min = vdupq_n_f32(I16_MIN_F);
            let max = vdupq_n_f32(I16_MAX_F);
            while i + 8 <= len {
                let a0 = vld1q_f32(src.as_ptr().add(i));
                let a1 = vld1q_f32(src.as_ptr().add(i + 4));
                let s0 = vmulq_f32(a0, scale);
                let s1 = vmulq_f32(a1, scale);
                let c0 = vmaxq_f32(vminq_f32(s0, max), min);
                let c1 = vmaxq_f32(vminq_f32(s1, max), min);
                let i0 = vcvtnq_s32_f32(c0);
                let i1 = vcvtnq_s32_f32(c1);
                vst1q_s16(dst.as_mut_ptr().add(i), vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
                i += 8;
            }
        }
        while i < len { dst[i] = f32_to_i16(src[i]); i += 1; }
    }

    pub fn convert_i16_to_f32_neon(src: &[i16], dst: &mut [f32]) {
        let len = src.len();
        let mut i = 0;
        unsafe {
            let inv_scale = vdupq_n_f32(INV_SCALE);
            while i + 8 <= len {
                let input = vld1q_s16(src.as_ptr().add(i) as *const i16);
                let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(input))), inv_scale);
                let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(input))), inv_scale);
                vst1q_f32(dst.as_mut_ptr().add(i), f0);
                vst1q_f32(dst.as_mut_ptr().add(i + 4), f1);
                i += 8;
            }
        }
        while i < len { dst[i] = i16_to_f32(src[i]); i += 1; }
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn convert_f32_to_i16_avx2(src: &[f32], dst: &mut [i16]) {
        let len = src.len();
        let mut i = 0;
        let scale = _mm256_set1_ps(SCALE);
        let min = _mm256_set1_ps(I16_MIN_F);
        let max = _mm256_set1_ps(I16_MAX_F);

        while i + 8 <= len {
            let s = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), scale);
            let c = _mm256_max_ps(_mm256_min_ps(s, max), min);
            let i32v = _mm256_cvtps_epi32(c);

            let lo = _mm256_castsi256_si128(i32v);
            let hi = _mm256_extracti128_si256(i32v, 1);
            let i16v = _mm_packs_epi32(lo, hi);
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, i16v);
            i += 8;
        }

        while i < len { dst[i] = f32_to_i16(src[i]); i += 1; }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn convert_i16_to_f32_avx2(src: &[i16], dst: &mut [f32]) {
        let len = src.len();
        let mut i = 0;
        let inv_scale = _mm256_set1_ps(INV_SCALE);

        while i + 8 <= len {
            let input = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            let i32v = _mm256_cvtepi16_epi32(input);
            let f = _mm256_mul_ps(_mm256_cvtepi32_ps(i32v), inv_scale);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), f);
            i += 8;
        }

        while i < len { dst[i] = i16_to_f32(src[i]); i += 1; }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn convert_f32_to_i16_sse2(src: &[f32], dst: &mut [i16]) {
        let len = src.len();
        let mut i = 0;
        let scale = _mm_set1_ps(SCALE);
        let min = _mm_set1_ps(I16_MIN_F);
        let max = _mm_set1_ps(I16_MAX_F);
        while i + 8 <= len {
            let s0 = _mm_mul_ps(_mm_loadu_ps(src.as_ptr().add(i)), scale);
            let s1 = _mm_mul_ps(_mm_loadu_ps(src.as_ptr().add(i + 4)), scale);
            let c0 = _mm_max_ps(_mm_min_ps(s0, max), min);
            let c1 = _mm_max_ps(_mm_min_ps(s1, max), min);
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i,
                _mm_packs_epi32(_mm_cvtps_epi32(c0), _mm_cvtps_epi32(c1)));
            i += 8;
        }
        while i < len { dst[i] = f32_to_i16(src[i]); i += 1; }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn convert_i16_to_f32_sse2(src: &[i16], dst: &mut [f32]) {
        let len = src.len();
        let mut i = 0;
        let inv_scale = _mm_set1_ps(INV_SCALE);
        while i + 8 <= len {
            let input = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            let lo = _mm_srai_epi32(_mm_unpacklo_epi16(input, input), 16);
            let hi = _mm_srai_epi32(_mm_unpackhi_epi16(input, input), 16);
            _mm_storeu_ps(dst.as_mut_ptr().add(i), _mm_mul_ps(_mm_cvtepi32_ps(lo), inv_scale));
            _mm_storeu_ps(dst.as_mut_ptr().add(i + 4), _mm_mul_ps(_mm_cvtepi32_ps(hi), inv_scale));
            i += 8;
        }
        while i < len { dst[i] = i16_to_f32(src[i]); i += 1; }
    }
}

#[inline]
pub fn convert_encoder_input(src: &[f32], dst: &mut [i16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")] { simd::convert_f32_to_i16_neon(src, dst); return; }
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") { unsafe { simd::convert_f32_to_i16_avx2(src, dst); return; } }
        if is_x86_feature_detected!(sse2) { unsafe { simd::convert_f32_to_i16_sse2(src, dst); return; } }
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) { *d = f32_to_i16(*s); }
}

#[inline]
pub fn convert_decoder_output(src: &[i16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")] { simd::convert_i16_to_f32_neon(src, dst); return; }
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") { unsafe { simd::convert_i16_to_f32_avx2(src, dst); return; } }
        if is_x86_feature_detected!(sse2) { unsafe { simd::convert_i16_to_f32_sse2(src, dst); return; } }
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) { *d = i16_to_f32(*s); }
}

#[inline]
pub fn i16_slice_to_f32(src: &[i16], dst: &mut [f32]) {
    convert_decoder_output(src, dst);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conversions() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), 32767);
        assert_eq!(f32_to_i16(-1.0), -32768);
        assert_eq!(f32_to_i16(0.5), 16384);
        assert_eq!(f32_to_i16(-0.5), -16384);
        assert_eq!(f32_to_i16(10.0), 32767); // saturate
        assert_eq!(f32_to_i16(-10.0), -32768); // saturate
        
        assert!((i16_to_f32(0) - 0.0).abs() < 1e-6);
        assert!((i16_to_f32(32767) - 0.99997).abs() < 1e-4);
        assert!((i16_to_f32(-32768) + 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_slice_conversions() {
        let f32_vals: Vec<f32> = (0..100).map(|i| (i as f32 / 50.0) - 1.0).collect();
        let mut i16_vals = vec![0i16; 100];
        convert_encoder_input(&f32_vals, &mut i16_vals);
        
        let mut f32_back = vec![0.0f32; 100];
        convert_decoder_output(&i16_vals, &mut f32_back);
        
        for (orig, back) in f32_vals.iter().zip(f32_back.iter()) {
            assert!((orig - back).abs() < 0.001, "Diff too large: {} vs {}", orig, back);
        }
    }
}
