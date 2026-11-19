//! Replays every reproducer of `regressions/manifest.tsv` and
//! `regressions/manifest.d/*.tsv` through the fuzz target bodies, as
//! `regressions/check.sh` does with the libFuzzer binaries: each input runs
//! in its own process (this test binary, started again with `OFZ_REPLAY`)
//! with the entry's environment, and the entry's pattern must be absent
//! from that process's output.
//!
//! - `test:<name>` entries are the opusref tests of that name; they run in
//!   `cargo test -p opusref`.
//! - `<target>@aarch64` entries run only on aarch64.
//! - `<target>@dnn` and `<target>@fixed` entries run only with `--features
//!   dnn` or `--features fixed`, and such a build runs only its own entries:
//!   the dnn build parses the decoder and encoder inputs differently and
//!   changes SILK output, the fixed build compares with fixed-point libopus.
//! - The `stack-buffer-overflow` pattern needs AddressSanitizer; without it
//!   the entry runs, but its pattern cannot appear.
//! - An entry's `parse` column is the hash its input parses to
//!   (`OFZ_PARSE`, `common::parse_only`); the test fails when the input
//!   parses to another one, as after a change of the target's input layout.
//!
//! UB precondition checks and overflow checks need debug assertions, as in
//! cargo-fuzz builds, so this test refuses to run without them.

use std::path::{Path, PathBuf};
use std::process::Command;

const TEST: &str = "reproducers_stay_fixed";

/// The reference build: `dnn`, `fixed` or `default`.
const BUILD: &str = if cfg!(feature = "dnn") {
    "dnn"
} else if cfg!(feature = "fixed") {
    "fixed"
} else {
    "default"
};

/// The build an entry's target needs, from its `@dnn` or `@fixed` suffix.
fn build(target: &str) -> &'static str {
    match target.split_once('@') {
        Some((_, "dnn")) => "dnn",
        Some((_, "fixed")) => "fixed",
        _ => "default",
    }
}

fn target(name: &str) -> fn(&[u8]) {
    match name {
        "api_args" => opus_rs_fuzz::api_args::fuzz,
        "diff_decoder" => opus_rs_fuzz::diff_decoder::fuzz,
        "diff_encoder" => opus_rs_fuzz::diff_encoder::fuzz,
        "diff_kernels" => opus_rs_fuzz::diff_kernels::fuzz,
        "diff_ms_encoder" => opus_rs_fuzz::diff_ms_encoder::fuzz,
        "diff_multistream" => opus_rs_fuzz::diff_multistream::fuzz,
        "diff_projection" => opus_rs_fuzz::diff_projection::fuzz,
        "diff_repacketizer" => opus_rs_fuzz::diff_repacketizer::fuzz,
        _ => panic!("unknown fuzz target {name}"),
    }
}

struct Entry {
    id: String,
    target: String,
    env: Vec<(String, String)>,
    pattern: String,
    /// The hash of the parsed input, for the targets that print one.
    parse: Option<String>,
    desc: String,
}

fn manifest(dir: &Path) -> Vec<Entry> {
    let mut files = vec![dir.join("manifest.tsv")];
    let mut more: Vec<PathBuf> = std::fs::read_dir(dir.join("manifest.d"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "tsv"))
        .collect();
    more.sort();
    files.extend(more);
    let mut entries = Vec::new();
    for f in files {
        for line in std::fs::read_to_string(&f).unwrap().lines() {
            let cols: Vec<&str> = line.split('\t').filter(|c| !c.is_empty()).collect();
            if cols.is_empty() || cols[0].starts_with('#') {
                continue;
            }
            assert!(cols.len() >= 5, "{}: short line {line:?}", f.display());
            let env = match cols[2] {
                "-" => Vec::new(),
                e => e
                    .split(' ')
                    .map(|kv| {
                        let (k, v) = kv.split_once('=').expect("VAR=value");
                        (k.to_string(), v.to_string())
                    })
                    .collect(),
            };
            entries.push(Entry {
                id: cols[0].to_string(),
                target: cols[1].to_string(),
                env,
                pattern: cols[3].to_string(),
                parse: (cols[4] != "-").then(|| cols[4].to_string()),
                desc: cols.get(5).unwrap_or(&"").to_string(),
            });
        }
    }
    entries
}

#[test]
fn reproducers_stay_fixed() {
    if let Ok(spec) = std::env::var("OFZ_REPLAY") {
        let (name, path) = spec.split_once(':').unwrap();
        target(name)(&std::fs::read(path).unwrap());
        return;
    }
    assert!(
        cfg!(debug_assertions),
        "the replay needs debug assertions and overflow checks, as cargo-fuzz builds have; run it without --release"
    );
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("regressions");
    let exe = std::env::current_exe().unwrap();
    let mut present = Vec::new();
    for e in manifest(&dir) {
        let skip = if e.target.starts_with("test:") {
            Some("opusref test")
        } else if e.target.ends_with("@aarch64") && !cfg!(target_arch = "aarch64") {
            Some("aarch64 only")
        } else if build(&e.target) != BUILD {
            Some(match build(&e.target) {
                "dnn" => "--features dnn only",
                "fixed" => "--features fixed only",
                _ => "default build only",
            })
        } else {
            None
        };
        if let Some(why) = skip {
            println!("{:<4} {:<8} {} ({why})", e.id, "skip", e.desc);
            continue;
        }
        let input = dir.join(format!("{}.bin", e.id));
        assert!(input.exists(), "{}: missing {}", e.id, input.display());
        let name = e.target.split('@').next().unwrap();
        if let Some(want) = &e.parse {
            let out = Command::new(&exe)
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env("OFZ_REPLAY", format!("{name}:{}", input.display()))
                .env("OFZ_PARSE", "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stderr);
            let got = text
                .lines()
                .find_map(|l| l.strip_prefix("parsed "))
                .unwrap_or("nothing");
            if got != want {
                println!("{:<4} {:<8} {}", e.id, "drift", e.desc);
                present.push(format!(
                    "{} ({}): the input parses to {got}, the manifest records {want}; \
                     re-encode the reproducer for the new input layout (fuzz/DIFFERENTIAL.md)",
                    e.id, e.desc
                ));
                continue;
            }
        }
        let out = Command::new(&exe)
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env("OFZ_REPLAY", format!("{name}:{}", input.display()))
            .envs(e.env.iter().map(|(k, v)| (k, v)))
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
        let r = if text.contains(&e.pattern) {
            "present"
        } else {
            "absent"
        };
        println!("{:<4} {r:<8} {}", e.id, e.desc);
        if r == "present" {
            present.push(format!("{} ({}): {}", e.id, e.desc, text.trim_end()));
        }
    }
    assert!(
        present.is_empty(),
        "reproducers that still reproduce or parse differently:\n{}",
        present.join("\n\n")
    );
}
