//! Builds libopus (main at f1afc4222a, `scripts/fetch-libopus.sh`; float,
//! scalar C only) as the reference.
//! `-ffp-contract=off` and no intrinsics/RTCD, so results are reproducible
//! and match a straightforward float port operation for operation. With the
//! feature `fixed` it builds the fixed-point libopus (FIXED_POINT, SILK from
//! silk/fixed/), the reference of opus-rs's SILK encoder. With the feature
//! `simd` it builds libopus as its configure does on x86-64: intrinsics and
//! RTCD, each instruction set's files with its flags, and GCC's default
//! floating-point contraction (fused multiply-adds in the `-mfma` files).
//! The patches in `patches/` apply to copies of the files they change, in
//! `OUT_DIR`; the source tree stays as fetched.
use std::{env, fs, path::Path, path::PathBuf, process::Command};

fn list(src: &PathBuf, mk: &str, var: &str) -> Vec<PathBuf> {
    let text = fs::read_to_string(src.join(mk)).unwrap();
    let mut out = Vec::new();
    let mut on = false;
    // `VAR = file` or `VAR = \` and one file per line, each but the last
    // ending in `\`.
    for line in text.lines() {
        let l = line.trim();
        let item = match l.strip_prefix(&format!("{var} =")) {
            Some(rest) => {
                on = true;
                rest.trim()
            }
            None if on => l,
            None => continue,
        };
        let file = item.trim_end_matches('\\').trim();
        if !file.is_empty() {
            let patched = patched_dir().join(file);
            out.push(if patched.exists() {
                patched
            } else {
                src.join(file)
            });
        }
        if !item.ends_with('\\') {
            break;
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

/// The copies of the source files that `patches/` changes.
fn patched_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap()).join("patched")
}

/// Applies `patches/*.patch` (in name order, `patch -p1`) to copies of the
/// files they change in [`patched_dir`]. The patches change `.c` files
/// only: a copied `.c` file still finds the original headers.
fn apply_patches(src: &Path) {
    let out = patched_dir();
    if out.exists() {
        fs::remove_dir_all(&out).unwrap();
    }
    fs::create_dir_all(&out).unwrap();
    let dir = Path::new(&manifest_dir()).join("patches");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut patches: Vec<PathBuf> = match fs::read_dir(&dir) {
        Ok(d) => d
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "patch"))
            .collect(),
        Err(_) => Vec::new(),
    };
    patches.sort();
    for p in &patches {
        println!("cargo:rerun-if-changed={}", p.display());
        for line in fs::read_to_string(p).unwrap().lines() {
            let Some(file) = line.strip_prefix("+++ b/") else {
                continue;
            };
            let file = file.split('\t').next().unwrap().trim();
            assert!(
                file.ends_with(".c"),
                "{}: a patch changes .c files only, not {file}",
                p.display()
            );
            let copy = out.join(file);
            if !copy.exists() {
                fs::create_dir_all(copy.parent().unwrap()).unwrap();
                fs::copy(src.join(file), &copy).unwrap();
            }
        }
        let ok = Command::new("patch")
            .args(["-p1", "--batch", "--forward", "--no-backup-if-mismatch", "-s", "-d"])
            .arg(&out)
            .arg("-i")
            .arg(p)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "{} does not apply", p.display());
    }
}

fn main() {
    let src = PathBuf::from(
        env::var("OPUSREF_SRC").unwrap_or_else(|_| format!("{}/../../libopus", manifest_dir())),
    );
    println!("cargo:rerun-if-env-changed=OPUSREF_SRC");
    apply_patches(&src);
    println!("cargo:rerun-if-changed=shim.c");
    let fixed = env::var_os("CARGO_FEATURE_FIXED").is_some();
    let silk = if fixed { "fixed" } else { "float" };
    let simd = env::var_os("CARGO_FEATURE_SIMD").is_some();
    let dnn = env::var_os("CARGO_FEATURE_DNN").is_some();
    assert!(
        !(dnn && fixed),
        "the features dnn and fixed exclude each other"
    );
    let mut base = cc::Build::new();
    // The patched copies first: `wrap/` includes its file by its path in
    // the source tree.
    base.include(patched_dir())
        .include(&src)
        .include(manifest_dir())
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
        .flag("-fno-fast-math")
        .flag("-fwrapv")
        .warnings(false);
    if simd {
        for d in X86_DEFINES {
            base.define(d, None);
        }
    } else {
        base.flag("-ffp-contract=off");
    }
    if fixed {
        base.define("FIXED_POINT", None);
    }
    if dnn {
        // libopus's `--enable-deep-plc --enable-dred --enable-osce`: these
        // change the CELT, SILK and Opus layers too.
        for d in DNN_DEFINES {
            base.define(d, None);
        }
        base.include(src.join("dnn"));
    }
    let mut b = base.clone();
    let mut lists = vec![
        ("celt_sources.mk", "CELT_SOURCES".to_owned()),
        ("silk_sources.mk", "SILK_SOURCES".to_owned()),
        (
            "silk_sources.mk",
            format!("SILK_SOURCES_{}", silk.to_uppercase()),
        ),
        ("opus_sources.mk", "OPUS_SOURCES".to_owned()),
        ("opus_sources.mk", "OPUS_SOURCES_FLOAT".to_owned()),
    ];
    if simd {
        // The baseline instruction sets: SSE and SSE2 are part of x86-64.
        for (mk, var) in [
            ("celt_sources.mk", "CELT_SOURCES_X86_RTCD"),
            ("celt_sources.mk", "CELT_SOURCES_SSE"),
            ("celt_sources.mk", "CELT_SOURCES_SSE2"),
            ("silk_sources.mk", "SILK_SOURCES_X86_RTCD"),
        ] {
            lists.push((mk, var.to_owned()));
        }
        b.objects(x86_objects(&src, &base, fixed, false));
    }
    for (mk, var) in &lists {
        // Files compiled through `wrap/`: the decoder files trace each
        // range-decoder call (trace.h); opus_encoder.c records the samples it
        // hands to SILK (silk_hook.h); x86cpu.c lets tests force the level
        // opus_select_arch() returns.
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
            "x86cpu.c",
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
    if dnn {
        b.objects(build_dnn(&src, simd, &base));
    }
    b.file("trace.c").file("shim.c");
    b.compile("opusref");
}

/// The configure defines of libopus on x86-64 with intrinsics and RTCD: SSE
/// and SSE2 presumed, SSE4.1 and AVX2 chosen at run time.
const X86_DEFINES: [&str; 8] = [
    "OPUS_HAVE_RTCD",
    "CPU_INFO_BY_ASM",
    "OPUS_X86_MAY_HAVE_SSE",
    "OPUS_X86_PRESUME_SSE",
    "OPUS_X86_MAY_HAVE_SSE2",
    "OPUS_X86_PRESUME_SSE2",
    "OPUS_X86_MAY_HAVE_SSE4_1",
    "OPUS_X86_MAY_HAVE_AVX2",
];

/// The SSE4.1 and AVX2 files (of the DNN code with `dnn`), each set built
/// with its configure flags (`X86_SSE4_1_CFLAGS`, `X86_AVX2_CFLAGS`).
fn x86_objects(src: &PathBuf, base: &cc::Build, fixed: bool, dnn: bool) -> Vec<PathBuf> {
    let sets: [(&[&str], Vec<(&str, &str)>); 2] = if dnn {
        [
            (
                &["-msse4.1"],
                vec![("lpcnet_sources.mk", "DNN_SOURCES_SSE4_1")],
            ),
            (
                &["-mavx", "-mfma", "-mavx2"],
                vec![("lpcnet_sources.mk", "DNN_SOURCES_AVX2")],
            ),
        ]
    } else {
        let silk = if fixed {
            ("silk_sources.mk", "SILK_SOURCES_FIXED_SSE4_1")
        } else {
            ("silk_sources.mk", "SILK_SOURCES_FLOAT_AVX2")
        };
        let mut sse4_1 = vec![
            ("celt_sources.mk", "CELT_SOURCES_SSE4_1"),
            ("silk_sources.mk", "SILK_SOURCES_SSE4_1"),
        ];
        let mut avx2 = vec![
            ("celt_sources.mk", "CELT_SOURCES_AVX2"),
            ("silk_sources.mk", "SILK_SOURCES_AVX2"),
        ];
        if fixed {
            sse4_1.push(silk)
        } else {
            avx2.push(silk)
        }
        [
            (&["-msse4.1"], sse4_1),
            (&["-mavx", "-mfma", "-mavx2"], avx2),
        ]
    };
    let mut objects = Vec::new();
    for (flags, lists) in sets {
        let mut b = base.clone();
        for flag in flags {
            b.flag(flag);
        }
        for (mk, var) in lists {
            b.files(list(src, mk, var));
        }
        objects.extend(b.compile_intermediates());
    }
    objects
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
/// lpcnet_sources.mk) with the model data. Without `simd`, `-U__SSE2__`
/// makes `dnn/vec.h` use its generic C kernels: x86-64 compilers always
/// define `__SSE2__`, which would select `vec_avx.h` (SSE intrinsics,
/// unsigned int8 inputs with `USE_SU_BIAS`), and `DISABLE_NEON` does the
/// same on aarch64, whose compilers always define `__ARM_NEON` (which would
/// select `vec_neon.h`). With `simd` the files build as
/// configure builds them, with the x86 kernels. A file with a
/// `wrap/dnn/<name>` counterpart is compiled through it, so tests can call
/// its static functions. The objects go into the one `opusref` archive.
fn build_dnn(src: &PathBuf, simd: bool, base: &cc::Build) -> Vec<PathBuf> {
    let mut d = cc::Build::new();
    let mut lists = vec!["DEEP_PLC_SOURCES", "DRED_SOURCES", "OSCE_SOURCES"];
    if simd {
        lists.extend(["DNN_SOURCES_X86_RTCD", "DNN_SOURCES_SSE2"]);
    }
    for var in lists {
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
    d.include(patched_dir())
        .include(src)
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
        .flag("-fno-fast-math")
        .flag("-fwrapv")
        .warnings(false);
    let mut objects = Vec::new();
    if simd {
        for def in X86_DEFINES {
            d.define(def, None);
        }
        let mut dnn_base = base.clone();
        dnn_base.include(src.join("silk/float"));
        objects = x86_objects(src, &dnn_base, false, true);
    } else {
        d.flag("-U__SSE2__")
            .define("DISABLE_NEON", None)
            .flag("-ffp-contract=off");
    }
    objects.extend(d.compile_intermediates());
    objects
}
