#![cfg(feature = "dnn")]
//! DRED of opus-rs (quantizer tables and schedule, Laplace coder, RDOVAE
//! encoder and decoder, `DREDEnc` and `OpusDRED`) against libopus: every
//! float must have the same bits and every payload the same bytes.
use opus_rs::dnn::builtin;
use opus_rs::dnn::dred::{
    DredEnc, OpusDred, coding, decoder, encoder, laplace, rdovae_dec, rdovae_enc, stats,
};
use opus_rs::range_coder::RangeCoder;
use opusref::dnn::{Rng, same_bits};
use opusref::dnn_dred as c;

#[test]
fn stats_tables_match() {
    unsafe {
        assert_eq!(
            stats::DRED_LATENT_QUANT_SCALES_Q8,
            c::dred_latent_quant_scales_q8
        );
        assert_eq!(stats::DRED_LATENT_DEAD_ZONE_Q8, c::dred_latent_dead_zone_q8);
        assert_eq!(stats::DRED_LATENT_R_Q8, c::dred_latent_r_q8);
        assert_eq!(stats::DRED_LATENT_P0_Q8, c::dred_latent_p0_q8);
        assert_eq!(
            stats::DRED_STATE_QUANT_SCALES_Q8,
            c::dred_state_quant_scales_q8
        );
        assert_eq!(stats::DRED_STATE_DEAD_ZONE_Q8, c::dred_state_dead_zone_q8);
        assert_eq!(stats::DRED_STATE_R_Q8, c::dred_state_r_q8);
        assert_eq!(stats::DRED_STATE_P0_Q8, c::dred_state_p0_q8);
    }
}

#[test]
fn compute_quantizer_matches() {
    for q0 in 0..16 {
        for dq in 0..8 {
            for qmax in q0..16 {
                for i in 0..60 {
                    let a = unsafe { c::ref_compute_quantizer(q0, dq, qmax, i) };
                    assert_eq!(
                        a,
                        coding::compute_quantizer(q0, dq, qmax, i),
                        "{q0} {dq} {qmax} {i}"
                    );
                }
            }
        }
    }
}

/// Random Laplace-coded symbols: mostly small values, some up to ±40, with
/// `p0` and `decay` as DRED makes them from its Q8 tables, or any Q15
/// value. `p0` stays below 32767: at 32767 the negative sign has zero
/// probability and coding it leaves a zero range, which hangs both coders.
fn laplace_case(r: &mut Rng, n: usize) -> (Vec<i32>, Vec<u16>, Vec<u16>) {
    let mut vals = Vec::new();
    let mut p0 = Vec::new();
    let mut decay = Vec::new();
    for _ in 0..n {
        let v = match r.below(4) {
            0 => 0,
            1 => r.below(5) as i32 - 2,
            2 => r.below(17) as i32 - 8,
            _ => r.below(81) as i32 - 40,
        };
        vals.push(v);
        if r.below(2) == 0 {
            p0.push(((1 + r.below(254)) << 7) as u16);
            decay.push(((1 + r.below(254)) << 7) as u16);
        } else {
            p0.push(1 + r.below(32766) as u16);
            decay.push(r.below(32768) as u16);
        }
    }
    (vals, p0, decay)
}

#[test]
fn laplace_p0_matches() {
    let mut r = Rng(11);
    for case in 0..400 {
        let n = 1 + r.below(200) as usize;
        let (vals, p0, decay) = laplace_case(&mut r, n);
        let cap = [8usize, 40, 200, 1000][case % 4];
        let cres = c::laplace_p0_encode(&vals, &p0, &decay, cap);
        let mut enc = RangeCoder::new_encoder(cap as u32);
        for i in 0..n {
            laplace::ec_laplace_encode_p0(&mut enc, vals[i], p0[i], decay[i]);
        }
        let tell = enc.tell();
        enc.done();
        let Some((cbuf, ctell)) = cres else {
            assert_ne!(enc.error, 0, "case {case}: C failed, Rust did not");
            continue;
        };
        assert_eq!(enc.error, 0, "case {case}");
        assert_eq!(ctell, tell, "case {case}");
        assert_eq!(cbuf, enc.buf[..cap].to_vec(), "case {case}");

        // Decode the payload, and also a corrupted copy of it.
        let mut bytes = cbuf.clone();
        for pass in 0..2 {
            let (cv, ct) = c::laplace_p0_decode(&p0, &decay, &bytes);
            let mut dec = RangeCoder::new_decoder(&bytes);
            let rv: Vec<i32> = (0..n)
                .map(|i| laplace::ec_laplace_decode_p0(&mut dec, p0[i], decay[i]))
                .collect();
            assert_eq!(cv, rv, "case {case} pass {pass}");
            assert_eq!(ct, dec.tell(), "case {case} pass {pass}");
            if pass == 0 {
                assert_eq!(cv, vals, "case {case}: round trip");
            }
            for b in bytes.iter_mut() {
                *b = r.next_u32() as u8;
            }
        }
    }
}

#[test]
fn filter_df2t_matches() {
    let mut r = Rng(12);
    for _ in 0..50 {
        let order = 1 + r.below(8) as usize;
        let len = 1 + r.below(500) as usize;
        let b: Vec<f32> = (0..order).map(|_| r.uni() * 0.2).collect();
        let a: Vec<f32> = (0..order).map(|_| r.uni() * 0.2).collect();
        let b0 = r.uni();
        let x: Vec<f32> = (0..len).map(|_| r.uni() * 30000.0).collect();
        let mut mem_c: Vec<f32> = (0..=order).map(|_| r.uni()).collect();
        let mut mem_r = mem_c.clone();
        let mut yc = vec![0f32; len];
        let mut yr = vec![0f32; len];
        unsafe {
            c::ref_filter_df2t(
                x.as_ptr(),
                yc.as_mut_ptr(),
                len as i32,
                b0,
                b.as_ptr(),
                a.as_ptr(),
                order as i32,
                mem_c.as_mut_ptr(),
            )
        };
        encoder::filter_df2t(&x, &mut yr, b0, &b, &a, &mut mem_r);
        assert_eq!(same_bits(&yc, &yr), None);
        assert_eq!(same_bits(&mem_c, &mem_r), None);
    }
}

fn rdovae_enc_state(s: &rdovae_enc::RdovaeEncState) -> (Vec<f32>, bool) {
    let v = [
        &s.gru1_state[..],
        &s.gru2_state,
        &s.gru3_state,
        &s.gru4_state,
        &s.gru5_state,
        &s.conv1_state,
        &s.conv2_state,
        &s.conv3_state,
        &s.conv4_state,
        &s.conv5_state,
    ]
    .concat();
    (v, s.initialized)
}

fn rdovae_dec_state(s: &rdovae_dec::RdovaeDecState) -> (Vec<f32>, bool) {
    let v = [
        &s.gru1_state[..],
        &s.gru2_state,
        &s.gru3_state,
        &s.gru4_state,
        &s.gru5_state,
        &s.conv1_state,
        &s.conv2_state,
        &s.conv3_state,
        &s.conv4_state,
        &s.conv5_state,
    ]
    .concat();
    (v, s.initialized)
}

fn assert_state_eq(a: &(Vec<f32>, bool), b: &(Vec<f32>, bool), what: &str) {
    assert_eq!(a.0.len(), b.0.len(), "{what}: state size");
    assert_eq!(same_bits(&a.0, &b.0), None, "{what}: state");
    assert_eq!(a.1, b.1, "{what}: initialized");
}

#[test]
fn rdovae_encoder_matches() {
    let model = builtin::dred_rdovae_enc().unwrap();
    for (seed, scale) in [(21u64, 1.0f32), (22, 4.0), (23, 0.05)] {
        let mut r = Rng(seed);
        let mut cst = c::RdovaeEnc::new();
        let mut rst = rdovae_enc::RdovaeEncState::default();
        assert_state_eq(&cst.state(), &rdovae_enc_state(&rst), "initial");
        for f in 0..300 {
            let input: [f32; 40] = core::array::from_fn(|_| r.uni() * scale);
            let (mut la, mut sa) = ([0f32; 25], [0f32; 50]);
            let (mut lb, mut sb) = ([0f32; 25], [0f32; 50]);
            cst.encode_dframe(&mut la, &mut sa, &input);
            rdovae_enc::dred_rdovae_encode_dframe(&mut rst, &model, &mut lb, &mut sb, &input);
            assert_eq!(same_bits(&la, &lb), None, "seed {seed} frame {f}: latents");
            assert_eq!(
                same_bits(&sa, &sb),
                None,
                "seed {seed} frame {f}: initial state"
            );
            assert_state_eq(
                &cst.state(),
                &rdovae_enc_state(&rst),
                &format!("seed {seed} frame {f}"),
            );
        }
    }
}

#[test]
fn rdovae_decoder_matches() {
    let model = builtin::dred_rdovae_dec().unwrap();
    for (seed, scale) in [(31u64, 1.0f32), (32, 6.0), (33, 0.1)] {
        let mut r = Rng(seed);
        let mut cst = c::RdovaeDec::new();
        let mut rst = rdovae_dec::RdovaeDecState::default();
        for f in 0..300 {
            if f % 60 == 0 {
                // A new initial state; the conv states keep their history
                // except the first, as in libopus.
                let init: [f32; 50] = core::array::from_fn(|_| r.uni() * scale);
                cst.init_states(&init);
                rdovae_dec::dred_rdovae_dec_init_states(&mut rst, &model, &init);
                assert_state_eq(
                    &cst.state(),
                    &rdovae_dec_state(&rst),
                    &format!("seed {seed} init {f}"),
                );
            }
            let mut z: [f32; 26] = core::array::from_fn(|_| r.uni() * scale);
            z[25] = (r.below(16) as f64 * 0.125 - 1.0) as f32;
            let mut qa = [0f32; 80];
            let mut qb = [0f32; 80];
            cst.decode_qframe(&mut qa, &z);
            rdovae_dec::dred_rdovae_decode_qframe(&mut rst, &model, &mut qb, &z);
            assert_eq!(same_bits(&qa, &qb), None, "seed {seed} frame {f}");
            assert_state_eq(
                &cst.state(),
                &rdovae_dec_state(&rst),
                &format!("seed {seed} frame {f}"),
            );
        }
        for nb in [0usize, 1, 2, 7, 26] {
            let init: [f32; 50] = core::array::from_fn(|_| r.uni() * scale);
            let lat: Vec<f32> = (0..26 * nb).map(|_| r.uni() * scale).collect();
            let a = cst.decode_all(&init, &lat, nb);
            let mut b = vec![0f32; 80 * nb];
            rdovae_dec::dred_rdovae_decode_all(&model, &mut b, &init, &lat, nb);
            assert_eq!(same_bits(&a, &b), None, "seed {seed} decode_all {nb}");
        }
    }
}

/// A speech-like signal at `fs` Hz in the ±1 range, `channels` interleaved:
/// harmonics of a gliding pitch with an envelope, noise, and silences.
fn speechlike(n: usize, fs: u32, channels: usize, seed: u64, gain: f32) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut phase = 0.0f64;
    let mut out = Vec::with_capacity(n * channels);
    for i in 0..n {
        let t = i as f64 / fs as f64;
        let f0 = 110.0 + 70.0 * (t * 1.3).sin() + 30.0 * (t * 5.1).sin();
        phase += 2.0 * std::f64::consts::PI * f0 / fs as f64;
        let env = ((t * 2.7).sin() * 0.5 + 0.5).powi(2);
        let gate = if ((t * 4.0) as usize) % 5 == 4 {
            0.0
        } else {
            1.0
        };
        let mut v = 0.0;
        for h in 1..12 {
            if f0 * h as f64 > 0.45 * fs as f64 {
                break;
            }
            v += (phase * h as f64).sin() / h as f64;
        }
        let s = (gate * env * 0.28 * v) as f32;
        for ch in 0..channels {
            let w = if ch == 0 { 1.0 } else { 0.7 };
            out.push(gain * (w * s + 0.01 * r.uni()));
        }
    }
    out
}

fn rust_fields(e: &DredEnc<'_>) -> c::DredEncFields {
    c::DredEncFields {
        ints: [
            1,
            e.fs,
            e.channels,
            e.input_buffer_fill,
            e.dred_offset,
            e.latent_offset,
            e.last_extra_dred_offset,
            e.latents_buffer_fill,
        ],
        input_buffer: e.input_buffer.to_vec(),
        latents_buffer: e.latents_buffer.to_vec(),
        state_buffer: e.state_buffer.to_vec(),
        resample_mem: e.resample_mem.to_vec(),
        rdovae: rdovae_enc_state(&e.rdovae_enc),
    }
}

fn assert_enc_eq(a: &c::DredEncFields, b: &c::DredEncFields, what: &str) {
    assert_eq!(a.ints, b.ints, "{what}: ints");
    assert_eq!(
        same_bits(&a.input_buffer, &b.input_buffer),
        None,
        "{what}: input_buffer"
    );
    assert_eq!(
        same_bits(&a.latents_buffer, &b.latents_buffer),
        None,
        "{what}: latents_buffer"
    );
    assert_eq!(
        same_bits(&a.state_buffer, &b.state_buffer),
        None,
        "{what}: state_buffer"
    );
    assert_eq!(
        same_bits(&a.resample_mem, &b.resample_mem),
        None,
        "{what}: resample_mem"
    );
    assert_state_eq(&a.rdovae, &b.rdovae, what);
}

/// The frame sizes (in 2.5 ms units) used in a run: fixed, or cycling
/// through every Opus frame size the DRED encoder sees (up to 60 ms).
fn frame_units(kind: usize, k: usize) -> usize {
    match kind {
        0 => 8,
        1 => 4,
        2 => [1, 2, 4, 8, 16, 24, 8, 2][k % 8],
        _ => 24,
    }
}

#[test]
fn compute_latents_matches() {
    for fs in [8000u32, 12000, 16000, 24000, 48000] {
        for channels in [1usize, 2] {
            for kind in 0..4 {
                let seed = fs as u64 + channels as u64 * 7 + kind as u64;
                let gain = [1.0f32, 3.0, 0.02, 1.0][kind];
                let sig = speechlike(fs as usize * 3, fs, channels, seed, gain);
                let extra_delay =
                    [0, fs as i32 / 400, fs as i32 / 250, fs as i32 * 13 / 1000][kind];
                let mut cenc = c::DredEnc::new(fs as i32, channels as i32);
                let mut renc = DredEnc::builtin(fs as i32, channels as i32).unwrap();
                let what = format!("{fs} Hz {channels} ch kind {kind}");
                assert_enc_eq(&cenc.fields(), &rust_fields(&renc), &format!("{what} init"));
                let mut pos = 0;
                let mut k = 0;
                loop {
                    let frame = frame_units(kind, k) * fs as usize / 400;
                    // libopus reads up to 2*frame values per channel pair
                    // (see `DredEnc::compute_latents`).
                    if (pos + 2 * frame) * channels > sig.len() {
                        break;
                    }
                    let pcm = &sig[pos * channels..];
                    cenc.compute_latents(pcm, frame, channels, extra_delay);
                    renc.compute_latents(pcm, frame, extra_delay);
                    assert_enc_eq(
                        &cenc.fields(),
                        &rust_fields(&renc),
                        &format!("{what} frame {k}"),
                    );
                    pos += frame;
                    k += 1;
                    if k == 70 {
                        cenc.reset();
                        renc.reset();
                        assert_enc_eq(
                            &cenc.fields(),
                            &rust_fields(&renc),
                            &format!("{what} reset"),
                        );
                    }
                }
            }
        }
    }
}

/// One step of opus_encoder.c's `activity_mem` update for a frame of
/// `units` 2.5 ms.
fn push_activity(mem: &mut [u8], units: usize, active: u8) {
    mem.copy_within(..416 - units, units);
    mem[..units].fill(active);
}

fn rust_dred(d: &OpusDred) -> c::DredFields {
    c::DredFields {
        fec_features: d.fec_features.to_vec(),
        state: d.state.to_vec(),
        latents: d.latents.to_vec(),
        nb_latents: d.nb_latents,
        process_stage: d.process_stage,
        dred_offset: d.dred_offset,
    }
}

fn assert_dred_eq(a: &c::DredFields, b: &c::DredFields, what: &str) {
    assert_eq!(
        same_bits(&a.fec_features, &b.fec_features),
        None,
        "{what}: fec_features"
    );
    assert_eq!(same_bits(&a.state, &b.state), None, "{what}: state");
    assert_eq!(same_bits(&a.latents, &b.latents), None, "{what}: latents");
    assert_eq!(
        (a.nb_latents, a.process_stage, a.dred_offset),
        (b.nb_latents, b.process_stage, b.dred_offset),
        "{what}: ints"
    );
}

/// Decodes `bytes` with both decoders (latents seeded with the same random
/// values), then runs the RDOVAE decoder; everything must match.
fn check_decode(
    bytes: &[u8],
    max_dred_features: i32,
    dred_frame_offset: i32,
    process: bool,
    r: &mut Rng,
    cmodel: &c::RdovaeDec,
    model: &opus_rs::dnn::models::dred_rdovae_dec::RdovaeDec<'_>,
    what: &str,
) {
    let mut cd = c::Dred::new();
    let mut rd = OpusDred::default();
    for (a, b) in cd.latents_mut().iter_mut().zip(rd.latents.iter_mut()) {
        *a = r.uni();
        *b = *a;
    }
    let na = cd.ec_decode(bytes, max_dred_features, dred_frame_offset);
    let nb = rd.ec_decode(bytes, max_dred_features, dred_frame_offset);
    assert_eq!(na, nb, "{what}: return");
    assert_dred_eq(&cd.fields(), &rust_dred(&rd), what);
    if process {
        cd.process(cmodel);
        assert!(rd.process(model));
        assert_dred_eq(&cd.fields(), &rust_dred(&rd), &format!("{what} processed"));
    }
}

#[test]
fn encode_silk_frame_and_ec_decode_match() {
    let dec_model = builtin::dred_rdovae_dec().unwrap();
    let cmodel = c::RdovaeDec::new();
    let mut r = Rng(41);
    let mut payloads = 0;
    let mut max_extra = 0;
    for (fs, channels) in [
        (16000u32, 1usize),
        (48000, 2),
        (8000, 1),
        (24000, 2),
        (12000, 1),
    ] {
        let sig = speechlike(fs as usize * 4, fs, channels, fs as u64, 1.0);
        let mut cenc = c::DredEnc::new(fs as i32, channels as i32);
        let mut renc = DredEnc::builtin(fs as i32, channels as i32).unwrap();
        // libopus reads up to 8 bytes past its 416-byte activity_mem; the
        // C side gets zeros there, the Rust side reads them as inactive.
        let mut act_c = vec![0u8; 424];
        let mut act_r = vec![0u8; 416];
        let mut pos = 0;
        let mut k = 0;
        let mut active = 1u8;
        loop {
            let units = [8usize, 4, 8, 16, 24, 2][(k / 10) % 6];
            let frame = units * fs as usize / 400;
            if (pos + 2 * frame) * channels > sig.len() {
                break;
            }
            let pcm = &sig[pos * channels..];
            let extra_delay = fs as i32 / 250;
            cenc.compute_latents(pcm, frame, channels, extra_delay);
            renc.compute_latents(pcm, frame, extra_delay);
            // Activity: runs with random switches, and one silence of over
            // a second (the encoder then skips most of its latents, and
            // libopus reads past its activity_mem).
            if r.below(if active == 1 { 12 } else { 6 }) == 0 {
                active ^= 1;
            }
            if (60..110).contains(&k) {
                active = 0;
            }
            push_activity(&mut act_c, units, active);
            push_activity(&mut act_r, units, active);
            let max_chunks = [26, 1, 3, 13, 26, 8][k % 6];
            let max_bytes = [998, 8, 20, 60, 200, 500][(k / 3) % 6];
            let q0 = r.below(16) as i32;
            let dq = if r.below(4) == 0 {
                0
            } else {
                r.below(8) as i32
            };
            let qmax = if q0 < 14 && dq > 0 {
                q0 + 1 + r.below((15 - q0) as u32) as i32
            } else {
                15
            };
            let a = cenc.encode_silk_frame(max_chunks, max_bytes, q0, dq, qmax, &mut act_c);
            let mut b = vec![0u8; max_bytes as usize];
            let n = renc.encode_silk_frame(&mut b, max_chunks, max_bytes, q0, dq, qmax, &act_r);
            b.truncate(n as usize);
            let what = format!(
                "{fs} Hz {channels} ch frame {k} ({max_chunks} {max_bytes} {q0} {dq} {qmax})"
            );
            assert_eq!(a, b, "{what}: payload");
            assert_enc_eq(&cenc.fields(), &rust_fields(&renc), &what);
            max_extra = max_extra.max(renc.last_extra_dred_offset);
            if !a.is_empty() {
                payloads += 1;
                for (mff, off, process) in [
                    (104, 0, true),
                    (2 + r.below(100) as i32, r.below(8) as i32 * 4, k % 4 == 0),
                ] {
                    check_decode(
                        &a,
                        mff,
                        off,
                        process,
                        &mut r,
                        &cmodel,
                        &dec_model,
                        &format!("{what} decode {mff} {off}"),
                    );
                }
                // Truncated payloads.
                let cut = r.below(a.len() as u32) as usize;
                check_decode(
                    &a[..cut],
                    104,
                    0,
                    k % 5 == 0,
                    &mut r,
                    &cmodel,
                    &dec_model,
                    &format!("{what} cut {cut}"),
                );
            }
            pos += frame;
            k += 1;
        }
    }
    assert!(payloads > 50, "only {payloads} payloads");
    // Skipped latents up to the end of the buffer (51 of 52).
    assert!(max_extra >= 51, "longest skip {max_extra}");
}

#[test]
fn ec_decode_random_bytes_matches() {
    let dec_model = builtin::dred_rdovae_dec().unwrap();
    let cmodel = c::RdovaeDec::new();
    let mut r = Rng(51);
    for case in 0..400 {
        let len = r.below(300) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| r.next_u32() as u8).collect();
        let mff = r.below(210) as i32 - 2;
        let off = r.below(12) as i32 * 4;
        check_decode(
            &bytes,
            mff,
            off,
            case % 8 == 0,
            &mut r,
            &cmodel,
            &dec_model,
            &format!("case {case}"),
        );
    }
}

#[test]
fn decoder_constants_match() {
    assert_eq!(
        decoder::DRED_FEC_FEATURES_SIZE,
        c::Dred::new().fields().fec_features.len()
    );
    assert_eq!(
        decoder::DRED_LATENTS_SIZE,
        c::Dred::new().fields().latents.len()
    );
}
