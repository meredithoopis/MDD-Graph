#!/usr/bin/env bash
set -euo pipefail
# https://drive.google.com/file/d/1kWY83KKv9ddlL-JWZO2RVCdc3RID6GbC/view

#   ./download_data.sh 1kWY83KKv9ddlL-JWZO2RVCdc3RID6GbC EN_MDD.zip

FILE_ID="${1:?Missing FILE_ID. Usage: $0 <FILE_ID> [OUTPUT_ZIP]}"
ZIP_NAME="${2:-data.zip}"

# Install gdown if missing
if ! command -v gdown >/dev/null 2>&1; then
  echo "[INFO] gdown not found. Installing..."
  python3 -m pip install --user -U gdown
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[INFO] Downloading zip from Google Drive ID: ${FILE_ID}"
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${ZIP_NAME}"

echo "[INFO] Unzipping: ${ZIP_NAME}"
unzip -o "${ZIP_NAME}"

echo "[INFO] Deleting zip: ${ZIP_NAME}"
rm -f "${ZIP_NAME}"

echo "[DONE] Downloaded, unzipped, and cleaned up."
