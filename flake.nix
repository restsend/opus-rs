{
  description = "opus-rs development shell and test environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs, ... }:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (
          system: f nixpkgs.legacyPackages.${system}
        );
      # libopus main at the revision opus-rs tracks.
      libopusSrc =
        pkgs:
        pkgs.fetchFromGitHub {
          owner = "xiph";
          repo = "opus";
          rev = "f1afc4222aff7c5456d27581ec618ae64158fbe1";
          hash = "sha256-mx6HoWA8t8NKm5AMs06kjGlmAJe9cEsQh61J1u+AM3I=";
        };
      # The RFC 8251 decoder test vectors.
      testvectors =
        pkgs:
        pkgs.runCommand "opus-testvectors-rfc8251" { } ''
          mkdir $out
          tar -xzf ${
            pkgs.fetchurl {
              url = "https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz";
              sha256 = "6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55";
            }
          } -C $out --strip-components=1
        '';
      # A drop-in libopus from capi/: libopus's C API, headers and
      # pkg-config file, with the cargo features of capi/Cargo.toml.
      mkLibopus =
        pkgs: features:
        pkgs.rustPlatform.buildRustPackage {
          pname = "libopus-opus-rs";
          version = "1.6.1";
          src = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./Cargo.toml
              ./src
              ./dnn-weights
              # Cargo.toml names this bench.
              ./benches/opus_bench.rs
              ./capi/Cargo.toml
              ./capi/Cargo.lock
              ./capi/build.rs
              ./capi/src
            ];
          };
          cargoRoot = "capi";
          buildAndTestSubdir = "capi";
          cargoLock.lockFile = ./capi/Cargo.lock;
          buildFeatures = features;
          nativeBuildInputs = pkgs.lib.optional pkgs.stdenv.hostPlatform.isDarwin pkgs.fixDarwinDylibNames;
          postInstall = ''
            if [ -e $out/lib/libopus.so ]; then
              mv $out/lib/libopus.so $out/lib/libopus.so.0
              ln -s libopus.so.0 $out/lib/libopus.so
            fi
            mkdir -p $out/include/opus $out/lib/pkgconfig
            cp ${libopusSrc pkgs}/include/opus{,_defines,_multistream,_projection,_types}.h $out/include/opus/
            # The system libraries a Rust static library links with.
            private=$(echo | rustc --crate-type staticlib --print native-static-libs -o $TMPDIR/empty.a - 2>&1 \
              | sed -n 's/.*native-static-libs: //p')
            cat > $out/lib/pkgconfig/opus.pc <<EOF
            prefix=$out
            libdir=$out/lib
            includedir=$out/include

            Name: Opus
            Description: Opus IETF audio codec (opus-rs, floating-point build)
            URL: https://opus-codec.org/
            Version: 1.6.1
            Libs: -L\''${libdir} -lopus
            Libs.private: $private
            Cflags: -I\''${includedir}/opus
            EOF
          '';
          meta = {
            description = "libopus's C API over opus-rs, a drop-in libopus";
            license = pkgs.lib.licenses.bsd3;
          };
        };
    in
    {
      packages = forAllSystems (pkgs: rec {
        libopus = mkLibopus pkgs [ ];
        # With libopus's neural features (--enable-deep-plc, --enable-dred,
        # --enable-osce).
        libopus-dnn = mkLibopus pkgs [
          "deep-plc"
          "dred"
          "osce"
        ];
        default = libopus;
      });

      # capi/test.sh against packages.libopus.
      checks = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          libopus =
            pkgs.runCommandCC "libopus-opus-rs-tests"
              {
                nativeBuildInputs = with pkgs; [
                  pkg-config
                  python3
                  glibc.bin
                ];
                buildInputs = [ pkgs.libopus ];
                OPUSREF_SRC = libopusSrc pkgs;
                OPUS_TESTVECTORS = testvectors pkgs;
                tests = pkgs.lib.fileset.toSource {
                  root = ./.;
                  fileset = pkgs.lib.fileset.unions [
                    ./capi/test.sh
                    ./capi/small_stack.c
                    ./fixtures/answer_16k.wav
                  ];
                };
              }
              ''
                bash $tests/capi/test.sh ${self.packages.${system}.libopus}/lib $TMPDIR/work
                touch $out
              '';
        }
      );

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            clippy
            rustfmt
            pkg-config
            cmake
            libopus
            python3
          ];
          # libopus with the neural model data autogen.sh names: the
          # differential reference fuzz/opusref builds and the source of
          # capi's test programs (scripts/fetch-libopus.sh fetches the same
          # without Nix).
          OPUSREF_SRC = pkgs.runCommand "libopus-f1afc4222a-src" { } ''
            cp -r ${libopusSrc pkgs} $out
            chmod -R u+w $out
            tar -xzf ${
              pkgs.fetchurl {
                url = "https://media.xiph.org/opus/models/opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz";
                sha256 = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e";
              }
            } -C $out --wildcards 'dnn/*_data.[ch]' 'dnn/dred_rdovae_constants.h'
          '';
          OPUS_TESTVECTORS = testvectors pkgs;
        };
      });
    };
}
