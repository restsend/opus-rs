//! Builds libopus 1.6.1 (float, scalar C only) as the reference.
//! `-ffp-contract=off` and no intrinsics/RTCD, so results are reproducible
//! and match a straightforward float port operation for operation.
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

fn main() {
    let src = PathBuf::from(
        env::var("OPUSREF_SRC")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../opus-1.6.1").into()),
    );
    println!("cargo:rerun-if-env-changed=OPUSREF_SRC");
    println!("cargo:rerun-if-changed=shim.c");
    let mut b = cc::Build::new();
    for (mk, var) in [
        ("celt_sources.mk", "CELT_SOURCES"),
        ("silk_sources.mk", "SILK_SOURCES"),
        ("silk_sources.mk", "SILK_SOURCES_FLOAT"),
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
    b.file("trace.c")
        .include(&src)
        .include(env!("CARGO_MANIFEST_DIR"))
        .file("shim.c")
        .include(src.join("include"))
        .include(src.join("celt"))
        .include(src.join("silk"))
        .include(src.join("silk/float"))
        .include(src.join("src"))
        .define("OPUS_BUILD", None)
        .define("VAR_ARRAYS", None)
        .define("HAVE_LRINTF", None)
        .define("HAVE_LRINT", None)
        .define("PACKAGE_VERSION", "\"1.6.1-ref\"")
        .opt_level(2)
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .flag("-fwrapv")
        .warnings(false);
    b.compile("opusref");
}
