//! Encodes one CELT-only (RestrictedLowDelay, 48 kHz) configuration with
//! opus-rs and libopus and prints, per frame, the first symbol the reference
//! decoder reads differently from the two packets.
//!
//! encdiff <channels> <ms> <kbps> <rc: 0 VBR, 1 CVBR, 2 CBR> <voice|noise|sine|silence+voice> [frames]
//!
//! With `OFZ_DUMP=1` and a reference built from `dump/instrument.py`, both
//! encoders also print their intermediate values (see dump/compare.py).
use opus_rs::{Application, OpusEncoder};
use opusref::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ch: usize = a[1].parse().unwrap();
    let ms: f32 = a[2].parse().unwrap();
    let kbps: i32 = a[3].parse().unwrap();
    let rc: i32 = a[4].parse().unwrap();
    let sig = a[5].clone();
    let frames: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(10);
    let fs = 48000usize;
    let frame = (fs as f32 * ms / 1000.0) as usize;
    let mut e = OpusEncoder::new(fs as i32, ch, Application::RestrictedLowDelay).unwrap();
    e.bitrate_bps = kbps * 1000;
    e.complexity = 10;
    e.use_cbr = rc == 2;
    e.set_vbr_constraint(rc == 1);
    let mut err = 0;
    let r = unsafe { opus_encoder_create(fs as i32, ch as i32, 2051, &mut err) };
    let d1 = unsafe { opus_decoder_create(fs as i32, ch as i32, &mut err) };
    let d2 = unsafe { opus_decoder_create(fs as i32, ch as i32, &mut err) };
    unsafe {
        opus_encoder_ctl(r, SET_BITRATE, kbps * 1000);
        opus_encoder_ctl(r, SET_COMPLEXITY, 10);
        opus_encoder_ctl(r, SET_VBR, (rc != 2) as i32);
        opus_encoder_ctl(r, SET_VBR_CONSTRAINT, (rc == 1) as i32);
    }
    let mut seed = 1u32;
    for f in 0..frames {
        let pcm: Vec<f32> = (0..frame * ch)
            .map(|i| {
                let t = (f * frame + i / ch) as f32 / fs as f32;
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                let voice = (0..20)
                    .map(|h| ((h + 1) as f32 * 140.0 * t * 6.2832).sin() / (h + 1) as f32)
                    .sum::<f32>()
                    * 0.2;
                match sig.as_str() {
                    "voice" => voice,
                    "noise" => noise * 0.5,
                    "quiet" => noise * 3e-6,
                    "sine" => (440.0 * t * 6.2832).sin() * 0.5,
                    "split" if i % ch == 0 => noise * 0.5,
                    "split" => voice,
                    "switch" => (if f < 4 { 440.0 } else { 3000.0 } * t * 6.2832).sin() * 0.5,
                    "grow" => noise * (0.0005 * 2f32.powf(t * 75.0)).min(0.5),
                    _ => if f < 10 { 0.0 } else { voice },
                }
            })
            .collect();
        let mut pa = [0u8; 1500];
        let mut pb = [0u8; 1500];
        let n = e.encode(&pcm, frame, &mut pa).unwrap();
        let m = unsafe { opus_encode_float(r, pcm.as_ptr(), frame as i32, pb.as_mut_ptr(), 1500) } as usize;
        let mut out = vec![0f32; frame * ch];
        let (_, sa) = trace(|| unsafe { opus_decode_float(d1, pa.as_ptr(), n as i32, out.as_mut_ptr(), frame as i32, 0) });
        let (_, sb) = trace(|| unsafe { opus_decode_float(d2, pb.as_ptr(), m as i32, out.as_mut_ptr(), frame as i32, 0) });
        if pa[..n] == pb[..m] {
            println!("frame {f}: identical ({n} bytes)");
            continue;
        }
        if std::env::var_os("ENCDIFF_PACKETS").is_some() {
            println!("opus-rs {:02x?}\nlibopus {:02x?}", &pa[..n], &pb[..m]);
        }
        let at = sa.iter().zip(&sb).position(|(x, y)| x != y);
        match at {
            Some(i) => {
                let ctx: Vec<String> = sb[i.saturating_sub(3)..i].iter().map(|s| format!("{}:{}={}", s.file, s.line, s.val)).collect();
                println!("frame {f}: sym {i} rs {}:{}={} ref {}:{}={} after [{}] ({n}/{m} bytes)", sa[i].file, sa[i].line, sa[i].val, sb[i].file, sb[i].line, sb[i].val, ctx.join(" "));
            }
            None => println!("frame {f}: same symbols, {n}/{m} bytes"),
        }
    }
}
