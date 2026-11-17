//! Builds libopus (main at f1afc4222a, `scripts/fetch-libopus.sh`; float,
//! scalar C only) as the reference.
//! `-ffp-contract=off` and no intrinsics/RTCD, so results are reproducible
//! and match a straightforward float port operation for operation. With the
//! feature `fixed` it builds the fixed-point libopus (FIXED_POINT, SILK from
//! silk/fixed/), the reference of opus-rs's SILK encoder.
use std::{env, fs, path::PathBuf};

fn list(src: &PathBuf, mk: &str, var: &str) -> Vec<PathBuf> {
    let text = fs::read_to_string(src.join(mk)).unwrap();
    let mut out = Vec::new();
    let mut on = false;
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with(&format!("{var} =")) {
            on = true;
            continue;
        }
        if on {
            if l.is_empty() {
                break;
            }
            out.push(src.join(l.trim_end_matches('\\').trim()));
        }
    }
    assert!(!out.is_empty(), "{var} empty");
    out
}

/// The package directory, read when the script runs: cargo reuses a
/// compiled build script for a copy of the package at another path, so a
/// compile-time `env!` would name the old path.
fn manifest_dir() -> String {
    env::var("CARGO_MANIFEST_DIR").unwrap()
}

fn main() {
    let src = PathBuf::from(
        env::var("OPUSREF_SRC").unwrap_or_else(|_| format!("{}/../../libopus", manifest_dir())),
    );
    println!("cargo:rerun-if-env-changed=OPUSREF_SRC");
    println!("cargo:rerun-if-changed=shim.c");
    let fixed = env::var_os("CARGO_FEATURE_FIXED").is_some();
    let silk = if fixed { "fixed" } else { "float" };
    let mut b = cc::Build::new();
    for (mk, var) in [
        ("celt_sources.mk", "CELT_SOURCES"),
        ("silk_sources.mk", "SILK_SOURCES"),
        (
            "silk_sources.mk",
            &format!("SILK_SOURCES_{}", silk.to_uppercase()),
        ),
        ("opus_sources.mk", "OPUS_SOURCES"),
        ("opus_sources.mk", "OPUS_SOURCES_FLOAT"),
    ] {
        // Files compiled through `wrap/`: the decoder files trace each
        // range-decoder call (trace.h); opus_encoder.c records the samples it
        // hands to SILK (silk_hook.h).
        let wrapped = [
            "celt_decoder.c",
            "bands.c",
            "quant_bands.c",
            "rate.c",
            "cwrs.c",
            "opus_decoder.c",
            "opus_encoder.c",
            "dec_API.c",
            "decode_indices.c",
            "decode_pulses.c",
            "code_signs.c",
            "shell_coder.c",
            "stereo_decode_pred.c",
        ];
        for f in list(&src, mk, var) {
            let name = f.file_name().unwrap().to_str().unwrap().to_owned();
            if wrapped.contains(&name.as_str()) {
                b.file(format!("wrap/{name}"));
            } else {
                b.file(f);
            }
        }
    }
    println!("cargo:rerun-if-changed=wrap");
    println!("cargo:rerun-if-changed=trace.c");
    println!("cargo:rerun-if-changed=trace.h");
    println!("cargo:rerun-if-changed=silk_hook.h");
    let dnn = env::var_os("CARGO_FEATURE_DNN").is_some();
    assert!(
        !(dnn && fixed),
        "the features dnn and fixed exclude each other"
    );
    if fixed {
        b.define("FIXED_POINT", None);
    }
    if dnn {
        b.objects(build_dnn(&src));
        // libopus's `--enable-deep-plc --enable-dred --enable-osce`: these
        // change the CELT, SILK and Opus layers too.
        for d in DNN_DEFINES {
            b.define(d, None);
        }
        b.include(src.join("dnn"));
    }
    b.file("trace.c")
        .include(&src)
        .include(manifest_dir())
        .file("shim.c")
        .include(src.join("include"))
        .include(src.join("celt"))
        .include(src.join("silk"))
        .include(src.join("silk").join(silk))
        .include(src.join("src"))
        .define("OPUS_BUILD", None)
        .define("VAR_ARRAYS", None)
        .define("HAVE_LRINTF", None)
        .define("HAVE_LRINT", None)
        .define("PACKAGE_VERSION", "\"f1afc4222a-ref\"")
        .opt_level(2)
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .flag("-fwrapv")
        .warnings(false);
    b.compile("opusref");
}

/// The configure defines of a libopus built with `--enable-deep-plc
/// --enable-dred --enable-osce` (which also sets ENABLE_OSCE_BWE).
/// DISABLE_DEBUG_FLOAT is libopus's default (`--enable-dnn-debug-float` is
/// off): the int8 layers keep only their int8 weights.
const DNN_DEFINES: [&str; 5] = [
    "ENABLE_DEEP_PLC",
    "ENABLE_DRED",
    "ENABLE_OSCE",
    "ENABLE_OSCE_BWE",
    "DISABLE_DEBUG_FLOAT",
];

/// Compiles libopus's `dnn/` sources (DEEP_PLC, DRED and OSCE lists of
/// lpcnet_sources.mk) with the model data. `-U__SSE2__` makes `dnn/vec.h`
/// use its generic C kernels: x86-64 compilers always define `__SSE2__`,
/// which would select `vec_avx.h` (SSE intrinsics, unsigned int8 inputs
/// with `USE_SU_BIAS`). A file with a `wrap/dnn/<name>` counterpart is
/// compiled through it, so tests can call its static functions. The
/// objects go into the one `opusref` archive.
fn build_dnn(src: &PathBuf) -> Vec<PathBuf> {
    let mut d = cc::Build::new();
    for var in ["DEEP_PLC_SOURCES", "DRED_SOURCES", "OSCE_SOURCES"] {
        for f in list(src, "lpcnet_sources.mk", var) {
            let name = f.file_name().unwrap().to_str().unwrap().to_owned();
            let wrapped = PathBuf::from(format!("wrap/dnn/{name}"));
            if wrapped.exists() {
                d.file(wrapped);
            } else {
                d.file(f);
            }
        }
    }
    println!("cargo:rerun-if-changed=wrap/dnn");
    // dnn_shim.c and one dnn_shim_<component>.c per component.
    let mut shims: Vec<PathBuf> = fs::read_dir(manifest_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_str().unwrap();
            n.starts_with("dnn_shim") && n.ends_with(".c")
        })
        .collect();
    shims.sort();
    for s in &shims {
        println!("cargo:rerun-if-changed={}", s.display());
        d.file(s);
    }
    println!("cargo:rerun-if-changed=.");
    for def in DNN_DEFINES {
        d.define(def, None);
    }
    d.include(src)
        .include(src.join("include"))
        .include(src.join("celt"))
        .include(src.join("silk"))
        .include(src.join("silk/float"))
        .include(src.join("src"))
        .include(src.join("dnn"))
        .define("OPUS_BUILD", None)
        .define("VAR_ARRAYS", None)
        .define("HAVE_LRINTF", None)
        .define("HAVE_LRINT", None)
        .opt_level(2)
        .flag("-U__SSE2__")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .flag("-fwrapv")
        .warnings(false);
    d.compile_intermediates()
}
