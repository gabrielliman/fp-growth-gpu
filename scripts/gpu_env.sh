#!/usr/bin/env bash
# Fonte para os scripts shell: fixa a GPU do slot PCIe x16 (barramento
# FPG_GPU_BUS) via CUDA_VISIBLE_DEVICES=<UUID>. Ver scripts/gpuutil.py.
FPG_GPU_BUS=${FPG_GPU_BUS:-00000000:01:00.0}
_uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v b="$FPG_GPU_BUS" 'tolower($1)==tolower(b){print $2}')
if [ -z "$_uuid" ]; then echo "GPU $FPG_GPU_BUS não encontrada" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="$_uuid"
