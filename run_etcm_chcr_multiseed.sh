#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/HDCTI_etcm_mention10_herb_only_seed2027.conf
  configs/HDCTI_etcm_mention10_chcr_seed2027.conf
  configs/HDCTI_etcm_mention10_herb_only_seed2028.conf
  configs/HDCTI_etcm_mention10_chcr_seed2028.conf
)

for config in "${configs[@]}"; do
  printf '\n===== Running %s =====\n' "$config"
  ./run_hdcti.sh "$config"
done
