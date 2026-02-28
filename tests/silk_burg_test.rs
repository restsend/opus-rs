#[cfg(test)]
mod tests {
    use opus_rs::silk::define::MAX_PREDICTION_POWER_GAIN;
    use opus_rs::silk::lpc_analysis::silk_burg_modified_fix;
    use std::f64::consts::PI;

    fn min_inv_gain_q30() -> i32 {
        // SILK_FIX_CONST(1.0 / MAX_PREDICTION_POWER_GAIN, 30)
        // MAX_PREDICTION_POWER_GAIN = 1e4 => 1/1e4 * 2^30 = 107374
        ((1.0 / MAX_PREDICTION_POWER_GAIN as f64) * (1u64 << 30) as f64) as i32
    }

    #[test]
    fn test_burg_sine() {
        let mut x = [0i16; 320];
        for i in 0..320 {
            x[i] = (10000.0 * (2.0 * PI * 440.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut res_nrg = 0i32;
        let mut res_nrg_q = 0i32;
        let mut a_q16 = [0i32; 24];

        silk_burg_modified_fix(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut a_q16,
            &x,
            min_inv_gain_q30(),
            80,
            4,
            16,
        );

        // C reference: res_nrg=19888 res_nrg_Q=-6
        // A_Q16=[129017,-65424,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        println!("TEST_SINE:");
        println!("  res_nrg={} res_nrg_q={}", res_nrg, res_nrg_q);
        print!("  A_Q16=[");
        for i in 0..16 {
            print!("{}", a_q16[i]);
            if i < 15 {
                print!(",");
            }
        }
        println!("]");

        // Check A_Q16 coefficients match within tolerance
        let c_a_q16 = [129017, -65424, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for i in 0..16 {
            let diff = (a_q16[i] - c_a_q16[i]).abs();
            assert!(
                diff <= 200,
                "A_Q16[{}] mismatch: rust={} c={} diff={}",
                i,
                a_q16[i],
                c_a_q16[i],
                diff
            );
        }
    }

    #[test]
    fn test_burg_noise() {
        let mut x = [0i16; 320];
        let mut seed: i32 = 12345;
        for i in 0..320 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            x[i] = (seed >> 16) as i16;
        }

        let mut res_nrg = 0i32;
        let mut res_nrg_q = 0i32;
        let mut a_q16 = [0i32; 24];

        silk_burg_modified_fix(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut a_q16,
            &x,
            min_inv_gain_q30(),
            80,
            4,
            16,
        );

        // C reference: res_nrg=703764586 res_nrg_Q=-7
        // A_Q16=[-3693,-638,-1336,-4866,4762,628,1119,7631,655,-6117,-2269,1005,1253,-8786,-8233,-4818]
        println!("TEST_NOISE:");
        println!("  res_nrg={} res_nrg_q={}", res_nrg, res_nrg_q);
        print!("  A_Q16=[");
        for i in 0..16 {
            print!("{}", a_q16[i]);
            if i < 15 {
                print!(",");
            }
        }
        println!("]");

        let c_a_q16 = [
            -3693, -638, -1336, -4866, 4762, 628, 1119, 7631, 655, -6117, -2269, 1005, 1253, -8786,
            -8233, -4818,
        ];
        for i in 0..16 {
            let diff = (a_q16[i] - c_a_q16[i]).abs();
            assert!(
                diff <= 200,
                "A_Q16[{}] mismatch: rust={} c={} diff={}",
                i,
                a_q16[i],
                c_a_q16[i],
                diff
            );
        }
    }

    #[test]
    fn test_burg_low_amp() {
        let mut x = [0i16; 320];
        for i in 0..320 {
            x[i] = (100.0 * (2.0 * PI * 300.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut res_nrg = 0i32;
        let mut res_nrg_q = 0i32;
        let mut a_q16 = [0i32; 24];

        silk_burg_modified_fix(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut a_q16,
            &x,
            min_inv_gain_q30(),
            80,
            4,
            16,
        );

        // C reference: res_nrg=15780 res_nrg_Q=7
        // A_Q16=[71536,26422,-10006,-24432,0,0,0,0,0,0,0,0,0,0,0,0]
        println!("TEST_LOW_AMP:");
        println!("  res_nrg={} res_nrg_q={}", res_nrg, res_nrg_q);
        print!("  A_Q16=[");
        for i in 0..16 {
            print!("{}", a_q16[i]);
            if i < 15 {
                print!(",");
            }
        }
        println!("]");

        let c_a_q16 = [
            71536, 26422, -10006, -24432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        for i in 0..16 {
            let diff = (a_q16[i] - c_a_q16[i]).abs();
            assert!(
                diff <= 200,
                "A_Q16[{}] mismatch: rust={} c={} diff={}",
                i,
                a_q16[i],
                c_a_q16[i],
                diff
            );
        }
    }

    #[test]
    fn test_burg_speech_like() {
        let mut x = [0i16; 320];
        let f0 = 150.0;
        for i in 0..320 {
            let mut s = 0.0f64;
            for h in 1..=10 {
                s += (5000.0 / h as f64) * (2.0 * PI * f0 * h as f64 * i as f64 / 16000.0).sin();
            }
            x[i] = s.clamp(-32768.0, 32767.0) as i16;
        }

        let mut res_nrg = 0i32;
        let mut res_nrg_q = 0i32;
        let mut a_q16 = [0i32; 24];

        silk_burg_modified_fix(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut a_q16,
            &x,
            min_inv_gain_q30(),
            80,
            4,
            16,
        );

        // C reference: res_nrg=7420 res_nrg_Q=-5
        // A_Q16=[161532,-75986,-66409,3552,41769,31578,-1546,-25322,-22967,-2039,16266,16982,1983,-13006,-9989,8971]
        println!("TEST_SPEECH_LIKE:");
        println!("  res_nrg={} res_nrg_q={}", res_nrg, res_nrg_q);
        print!("  A_Q16=[");
        for i in 0..16 {
            print!("{}", a_q16[i]);
            if i < 15 {
                print!(",");
            }
        }
        println!("]");

        let c_a_q16 = [
            161532, -75986, -66409, 3552, 41769, 31578, -1546, -25322, -22967, -2039, 16266, 16982,
            1983, -13006, -9989, 8971,
        ];
        for i in 0..16 {
            let diff = (a_q16[i] - c_a_q16[i]).abs();
            assert!(
                diff <= 200,
                "A_Q16[{}] mismatch: rust={} c={} diff={}",
                i,
                a_q16[i],
                c_a_q16[i],
                diff
            );
        }
    }

    #[test]
    fn test_burg_order10() {
        let mut x = [0i16; 320];
        for i in 0..320 {
            x[i] = (10000.0 * (2.0 * PI * 440.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut res_nrg = 0i32;
        let mut res_nrg_q = 0i32;
        let mut a_q16 = [0i32; 24];

        silk_burg_modified_fix(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut a_q16,
            &x,
            min_inv_gain_q30(),
            80,
            4,
            10,
        );

        // C reference: res_nrg=21672 res_nrg_Q=-6
        // A_Q16=[129017,-65424,0,0,0,0,0,0,0,0]
        println!("TEST_ORDER10:");
        println!("  res_nrg={} res_nrg_q={}", res_nrg, res_nrg_q);
        print!("  A_Q16=[");
        for i in 0..10 {
            print!("{}", a_q16[i]);
            if i < 9 {
                print!(",");
            }
        }
        println!("]");

        let c_a_q16 = [129017, -65424, 0, 0, 0, 0, 0, 0, 0, 0];
        for i in 0..10 {
            let diff = (a_q16[i] - c_a_q16[i]).abs();
            assert!(
                diff <= 200,
                "A_Q16[{}] mismatch: rust={} c={} diff={}",
                i,
                a_q16[i],
                c_a_q16[i],
                diff
            );
        }
    }
}
