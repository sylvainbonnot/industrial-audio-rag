#!/usr/bin/env bash
# Download & unzip DCASE 2024 Task‑2 dev set
# Usage: bash scripts/get_dcase24.sh [TARGET_DIR]
# Default TARGET_DIR = Data/Dcase

set -e
TARGET_DIR=${1:-Data/Dcase}
URL_BASE="https://zenodo.org/records/10902294/files"
FILES=(
  dev_bearing.zip
  dev_fan.zip
  dev_gearbox.zip
  dev_slider.zip
  dev_toycar.zip
  dev_toytrain.zip
  dev_valve.zip
)

echo "Target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

for file in "${FILES[@]}"; do
  echo "-> Downloading $file"
  wget -q --show-progress "$URL_BASE/$file" -O "$TARGET_DIR/$file"
  echo "   Extracting $file"
  unzip -q "$TARGET_DIR/$file" -d "$TARGET_DIR"
  rm "$TARGET_DIR/$file"
done

echo "✅ All machine‑type zips downloaded and extracted."
echo "   WAVs are under: $TARGET_DIR/<machine_type>/"
