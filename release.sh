#!/usr/bin/env bash
# One extension zip per platform with the runtime library and the model inside.
# Usage: MODEL_DIR=... RUNTIME_DIR=... ./release.sh macos-arm64=path/libsprongmotion.dylib linux-x64=... windows-x64=...
# MODEL_DIR: SprongMotion-14M-v1.0-Q8_0.gguf, LICENSE, README.md (the model card), NOTICE. RUNTIME_DIR: the runtime's LICENSE and NOTICE.
set -euo pipefail
cd "$(dirname "$0")"
BLENDER=${BLENDER:-blender} OUT=${OUT:-dist}
model=${MODEL_DIR:?set MODEL_DIR} runtime=${RUNTIME_DIR:?set RUNTIME_DIR}
version=$(sed -n 's/^version = "\(.*\)"/\1/p' blender_manifest.toml)
missing=0
for f in "$model"/{SprongMotion-14M-v1.0-Q8_0.gguf,LICENSE,README.md,NOTICE} "$runtime"/{LICENSE,NOTICE}; do
  [ -f "$f" ] || { echo "missing: $f"; missing=1; }
done
[ $missing = 0 ] || exit 1
for pair in "$@"; do
  p=${pair%%=*} lib=${pair#*=}
  [ -f "$lib" ] || { echo "skipped $p: no library at $lib"; missing=1; continue; }
  s=$OUT/stage/$p
  rm -rf "$s" && mkdir -p "$s/lib/$p" "$s/model" "$s/licenses"
  cp ./*.py "$s/" && cp -R wordlists "$s/"
  awk -v p="$p" '{print} /^blender_version_min/ {print "platforms = [\"" p "\"]"}' blender_manifest.toml \
    > "$s/blender_manifest.toml"
  cp "$lib" "$s/lib/$p/libsprongmotion.${lib##*.}"
  cp "$model/SprongMotion-14M-v1.0-Q8_0.gguf" "$s/model/"
  cp LICENSE "$s/licenses/ADDON-GPL-3.0-or-later.txt"
  cp "$runtime/LICENSE" "$s/licenses/RUNTIME-Apache-2.0.txt"
  cp "$runtime/NOTICE" "$s/licenses/RUNTIME-NOTICE.txt"
  cp "$model/LICENSE" "$s/licenses/MODEL-Apache-2.0.txt"
  cp "$model/README.md" "$s/licenses/MODEL_CARD.md"
  cp "$model/NOTICE" "$s/licenses/MODEL-NOTICE.txt"
  "$BLENDER" --factory-startup --command extension build --source-dir "$s" \
    --output-filepath "$OUT/sprongmotion-$version-$p.zip"
done
rm -rf "$OUT/stage"
exit $missing
