{
  description = "opus-rs development shell and test environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (
          system: f nixpkgs.legacyPackages.${system}
        );
    in
    {
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
          ];
          # libopus main at the revision opus-rs tracks, with the neural model
          # data autogen.sh names: the differential reference fuzz/opusref
          # builds (scripts/fetch-libopus.sh fetches the same without Nix).
          OPUSREF_SRC = pkgs.runCommand "libopus-f1afc4222a-src" { } ''
            cp -r ${
              pkgs.fetchFromGitHub {
                owner = "xiph";
                repo = "opus";
                rev = "f1afc4222aff7c5456d27581ec618ae64158fbe1";
                hash = "sha256-mx6HoWA8t8NKm5AMs06kjGlmAJe9cEsQh61J1u+AM3I=";
              }
            } $out
            chmod -R u+w $out
            tar -xzf ${
              pkgs.fetchurl {
                url = "https://media.xiph.org/opus/models/opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz";
                sha256 = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e";
              }
            } -C $out --wildcards 'dnn/*_data.[ch]' 'dnn/dred_rdovae_constants.h'
          '';
        };
      });
    };
}
