fn main() {
    // Same soname as libopus, so the .so can stand in for the system one.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libopus.so.0");
    }
}
