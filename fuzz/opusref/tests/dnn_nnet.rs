//! The layer functions of opus-rs (`dnn::nnet`) against libopus's generic
//! C kernels on random layers and inputs: every output must have the same
//! bits.
#![cfg(feature = "dnn")]
use opus_rs::dnn::nnet::{self as rs, Activation};
use opus_rs::dnn::weights::{F32s, I32s};
use opusref::dnn::{self as c, Rng, same_bits};

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A random layer, kept in the C types and as blob bytes.
struct Layer {
    nin: usize,
    nout: usize,
    bias: Option<Vec<f32>>,
    w8: Option<Vec<i8>>,
    fw: Option<Vec<f32>>,
    idx: Option<Vec<i32>>,
    diag: Option<Vec<f32>>,
    scale: Option<Vec<f32>>,
    b: [Option<Vec<u8>>; 6],
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Float,
    FloatSparse,
    Int8,
    Int8Sparse,
    Empty,
}

impl Layer {
    fn new(r: &mut Rng, nin: usize, nout: usize, kind: Kind, bias: bool, diag: bool) -> Layer {
        let vecf =
            |r: &mut Rng, n: usize, s: f32| (0..n).map(|_| r.uni() * s).collect::<Vec<f32>>();
        let mut idx = None;
        let mut blocks = 0;
        if matches!(kind, Kind::FloatSparse | Kind::Int8Sparse) {
            let mut v = Vec::new();
            for _ in 0..nout / 8 {
                let cols: Vec<i32> = (0..nin / 4)
                    .filter(|_| r.below(3) != 0)
                    .map(|c| 4 * c as i32)
                    .collect();
                v.push(cols.len() as i32);
                blocks += cols.len();
                v.extend(cols);
            }
            idx = Some(v);
        }
        let fw = match kind {
            Kind::Float => Some(vecf(r, nin * nout, 0.5)),
            Kind::FloatSparse => Some(vecf(r, 32 * blocks, 0.5)),
            _ => None,
        };
        let w8 = match kind {
            Kind::Int8 => Some(
                (0..nin * nout)
                    .map(|_| (r.below(255) as i32 - 127) as i8)
                    .collect::<Vec<i8>>(),
            ),
            Kind::Int8Sparse => Some(
                (0..32 * blocks)
                    .map(|_| (r.below(255) as i32 - 127) as i8)
                    .collect(),
            ),
            _ => None,
        };
        let scale = w8.as_ref().map(|_| vecf(r, nout, 1.0 / 256.0));
        let bias = bias.then(|| vecf(r, nout, 1.0));
        let diag = diag.then(|| vecf(r, nout, 0.5));
        let b = [
            bias.as_deref().map(bytes_f32),
            w8.as_ref().map(|w| w.iter().map(|&x| x as u8).collect()),
            fw.as_deref().map(bytes_f32),
            idx.as_deref().map(bytes_i32),
            diag.as_deref().map(bytes_f32),
            scale.as_deref().map(bytes_f32),
        ];
        Layer {
            nin,
            nout,
            bias,
            w8,
            fw,
            idx,
            diag,
            scale,
            b,
        }
    }

    fn c(&self) -> c::LinearLayer {
        fn p<T>(v: &Option<Vec<T>>) -> *const T {
            v.as_ref().map_or(core::ptr::null(), |v| v.as_ptr())
        }
        c::LinearLayer {
            bias: p(&self.bias),
            subias: core::ptr::null(),
            weights: p(&self.w8),
            float_weights: p(&self.fw),
            weights_idx: p(&self.idx),
            diag: p(&self.diag),
            scale: p(&self.scale),
            nb_inputs: self.nin as i32,
            nb_outputs: self.nout as i32,
        }
    }

    fn rs(&self) -> rs::LinearLayer<'_> {
        rs::LinearLayer {
            bias: self.b[0].as_deref().map(F32s::new),
            subias: None,
            weights: self.b[1].as_deref(),
            float_weights: self.b[2].as_deref().map(F32s::new),
            weights_idx: self.b[3].as_deref().map(I32s::new),
            diag: self.b[4].as_deref().map(F32s::new),
            scale: self.b[5].as_deref().map(F32s::new),
            nb_inputs: self.nin,
            nb_outputs: self.nout,
        }
    }
}

const ACTS: [Activation; 7] = [
    Activation::Linear,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Relu,
    Activation::Softmax,
    Activation::Swish,
    Activation::Exp,
];

fn input(r: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| r.uni() * scale).collect();
    // Values that int8 quantization wraps, and special values.
    if n > 8 && r.below(4) == 0 {
        v[r.below(n as u32) as usize] = 1.7;
        v[r.below(n as u32) as usize] = -3.2;
        v[r.below(n as u32) as usize] = 0.0039370;
        v[r.below(n as u32) as usize] = -0.0;
    }
    v
}

#[test]
fn activations_match() {
    let mut r = Rng(0x1234);
    let mut x: Vec<f32> = (0..4000).map(|_| r.uni() * 40.0).collect();
    x.extend([
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
        -1e30,
        88.0,
        -88.0,
        1e-40,
    ]);
    for act in ACTS {
        let mut a = vec![0f32; x.len()];
        let mut b = vec![0f32; x.len()];
        unsafe {
            c::ref_compute_activation(a.as_mut_ptr(), x.as_ptr(), x.len() as i32, act as i32)
        };
        rs::compute_activation(&mut b, &x, x.len(), act);
        assert_eq!(same_bits(&a, &b), None, "{act:?}");
        let mut d = x.clone();
        rs::compute_activation_inplace(&mut d, act);
        assert_eq!(same_bits(&a, &d), None, "{act:?} in place");
    }
}

#[test]
fn linear_and_dense_match() {
    let mut r = Rng(0x5678);
    for it in 0..400 {
        let kind = [
            Kind::Float,
            Kind::FloatSparse,
            Kind::Int8,
            Kind::Int8Sparse,
            Kind::Empty,
        ][it % 5];
        let nin = 4 * (1 + r.below(64) as usize);
        let nout = if kind == Kind::Float && r.below(2) == 0 {
            1 + r.below(100) as usize
        } else {
            8 * (1 + r.below(24) as usize)
        };
        let diag = false;
        let with_bias = r.below(4) != 0;
        let l = Layer::new(&mut r, nin, nout, kind, with_bias, diag);
        let amp = if r.below(2) == 0 { 1.0 } else { 4.0 };
        let x = input(&mut r, nin, amp);
        let act = ACTS[r.below(7) as usize];
        let mut a = vec![0f32; nout];
        let mut b = vec![0f32; nout];
        unsafe { c::ref_compute_generic_dense(&l.c(), a.as_mut_ptr(), x.as_ptr(), act as i32) };
        rs::compute_generic_dense(&l.rs(), &mut b, &x, act);
        assert_eq!(same_bits(&a, &b), None, "iteration {it}");
    }
}

#[test]
fn diag_gru_glu_match() {
    let mut r = Rng(0x9abc);
    for it in 0..200 {
        let kind = [Kind::Float, Kind::FloatSparse, Kind::Int8, Kind::Int8Sparse][it % 4];
        let n = 8 * (1 + r.below(24) as usize);
        let nin = 4 * (1 + r.below(48) as usize);
        let inp = Layer::new(&mut r, nin, 3 * n, kind, true, false);
        let rkind = [Kind::Float, Kind::Int8, Kind::Int8Sparse][it % 3];
        let rec = Layer::new(&mut r, n, 3 * n, rkind, true, it % 5 == 0);
        let x = input(&mut r, nin, 1.0);
        let st: Vec<f32> = input(&mut r, n, 1.0);
        let (mut sa, mut sb) = (st.clone(), st);
        for _ in 0..3 {
            unsafe { c::ref_compute_generic_gru(&inp.c(), &rec.c(), sa.as_mut_ptr(), x.as_ptr()) };
            rs::compute_generic_gru(&inp.rs(), &rec.rs(), &mut sb, &x);
            assert_eq!(same_bits(&sa, &sb), None, "gru iteration {it}");
        }
        let glu = Layer::new(&mut r, n, n, kind, true, false);
        let y = input(&mut r, n, 2.0);
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        unsafe { c::ref_compute_glu(&glu.c(), a.as_mut_ptr(), y.as_ptr()) };
        rs::compute_glu(&glu.rs(), &mut b, &y);
        assert_eq!(same_bits(&a, &b), None, "glu iteration {it}");
        let mut d = y.clone();
        rs::compute_glu_inplace(&glu.rs(), &mut d);
        assert_eq!(same_bits(&a, &d), None, "glu in place iteration {it}");
    }
}

#[test]
fn conv1d_match() {
    let mut r = Rng(0xdef0);
    for it in 0..200 {
        let kind = [Kind::Float, Kind::Int8, Kind::Int8Sparse, Kind::FloatSparse][it % 4];
        let input_size = 4 * (1 + r.below(16) as usize);
        // libopus's dilation path needs ksize >= 2 (it copies a negative count
        // otherwise).
        let ksize = 1 + r.below(4) as usize;
        let dilation = if it % 2 == 0 || ksize == 1 {
            1
        } else {
            1 + r.below(3) as usize
        };
        let nin = input_size * ksize;
        let nout = 8 * (1 + r.below(12) as usize);
        let l = Layer::new(&mut r, nin, nout, kind, true, false);
        let act = ACTS[r.below(7) as usize];
        let memlen = (nin - input_size) * dilation.max(1);
        let mut ma: Vec<f32> = input(&mut r, memlen.max(1), 1.0);
        let mut mb = ma.clone();
        for _ in 0..4 {
            let x = input(&mut r, input_size, 1.0);
            let mut a = vec![0f32; nout];
            let mut b = vec![0f32; nout];
            unsafe {
                c::ref_compute_generic_conv1d_dilation(
                    &l.c(),
                    a.as_mut_ptr(),
                    ma.as_mut_ptr(),
                    x.as_ptr(),
                    input_size as i32,
                    dilation as i32,
                    act as i32,
                )
            };
            rs::compute_generic_conv1d_dilation(
                &l.rs(),
                &mut b,
                &mut mb,
                &x,
                input_size,
                dilation,
                act,
            );
            assert_eq!(same_bits(&a, &b), None, "conv1d iteration {it}");
            assert_eq!(same_bits(&ma, &mb), None, "conv1d memory iteration {it}");
            if dilation == 1 {
                unsafe {
                    c::ref_compute_generic_conv1d(
                        &l.c(),
                        a.as_mut_ptr(),
                        ma.as_mut_ptr(),
                        x.as_ptr(),
                        input_size as i32,
                        act as i32,
                    )
                };
                rs::compute_generic_conv1d(&l.rs(), &mut b, &mut mb, &x, input_size, act);
                assert_eq!(same_bits(&a, &b), None, "conv1d iteration {it}");
                assert_eq!(same_bits(&ma, &mb), None, "conv1d memory iteration {it}");
            }
        }
    }
}

#[test]
fn conv2d_match() {
    let mut r = Rng(0x1357);
    for it in 0..60 {
        let (ktime, kheight) = if it % 2 == 0 {
            (3, 3)
        } else {
            (1 + r.below(3) as usize, 1 + r.below(5) as usize)
        };
        let inc = 1 + r.below(4) as usize;
        let outc = 1 + r.below(4) as usize;
        let height = 8 + r.below(60) as usize;
        let hstride = height + r.below(3) as usize;
        let w: Vec<f32> = (0..inc * outc * ktime * kheight).map(|_| r.uni()).collect();
        let bias: Vec<f32> = (0..outc).map(|_| r.uni()).collect();
        let (wb, bb) = (bytes_f32(&w), bytes_f32(&bias));
        let cl = c::Conv2dLayer {
            bias: bias.as_ptr(),
            float_weights: w.as_ptr(),
            in_channels: inc as i32,
            out_channels: outc as i32,
            ktime: ktime as i32,
            kheight: kheight as i32,
        };
        let rl = rs::Conv2dLayer {
            bias: Some(F32s::new(&bb)),
            float_weights: Some(F32s::new(&wb)),
            in_channels: inc,
            out_channels: outc,
            ktime,
            kheight,
        };
        let ts = inc * (height + kheight - 1);
        let mut ma: Vec<f32> = input(&mut r, (ktime - 1) * ts + 1, 1.0);
        let mut mb = ma.clone();
        let act = ACTS[r.below(7) as usize];
        for _ in 0..3 {
            let x = input(&mut r, ts, 1.0);
            let mut a = vec![0f32; outc * hstride];
            let mut b = vec![0f32; outc * hstride];
            unsafe {
                c::ref_compute_conv2d(
                    &cl,
                    a.as_mut_ptr(),
                    ma.as_mut_ptr(),
                    x.as_ptr(),
                    height as i32,
                    hstride as i32,
                    act as i32,
                )
            };
            rs::compute_conv2d(&rl, &mut b, &mut mb, &x, height, hstride, act);
            assert_eq!(same_bits(&a, &b), None, "conv2d iteration {it}");
            assert_eq!(same_bits(&ma, &mb), None, "conv2d memory iteration {it}");
        }
    }
}
