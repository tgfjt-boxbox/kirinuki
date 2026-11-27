#!/bin/bash
# Download @litertjs/core and @litertjs/wasm-utils (Apache-2.0) into bench/browser/vendor/litertjs/ and verify the npm integrity hashes.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=bench/browser/vendor/litertjs
mkdir -p "$dest"

while read -r name version integrity; do
  tgz="$dest/$name-$version.tgz"
  [ -f "$tgz" ] || curl -fsSL -o "$tgz" "https://registry.npmjs.org/@litertjs/$name/-/$name-$version.tgz"
  got="sha512-$(openssl dgst -sha512 -binary "$tgz" | base64)"
  [ "$got" = "$integrity" ] || { echo "integrity mismatch: $tgz" >&2; exit 1; }
  rm -rf "${dest:?}/$name"
  mkdir -p "$dest/$name"
  tar xzf "$tgz" -C "$dest/$name" --strip-components 1
  echo "ok  $name $version"
done <<'LIST'
core 2.6.0 sha512-TH6/5zLq57YzyZH6q+vO8be1SvTH76uDFe+CiL0wWtRNTUDhc2GAWm2YCohhnyt4fPvefSNztg5PKFZGAYL0cw==
wasm-utils 2.6.0 sha512-rxARtl8owf+aPV2TsMiAClYDBaEDemML3PjrzBxIpGCvv6POswR7068Kt0sIkrK5XvKkY2Z+JLcm3a9CHmvuUg==
LIST
