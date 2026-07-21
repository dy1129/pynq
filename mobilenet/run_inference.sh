#!/bin/bash
# PYNQ inference wrapper — sets up all PYNQ env vars then runs inference.py
#
# Usage on PYNQ:
#   sudo bash /home/xilinx/jupyter_notebooks/mobilenet/run_inference.sh
#
# Usage from PC (one-liner):
#   ssh xilinx@PYNQ "sudo bash /home/xilinx/jupyter_notebooks/mobilenet/run_inference.sh"

# === Required PYNQ environment variables ===
export XILINX_XRT=/usr                                # ⭐ Critical for device detection
export BOARD=Pynq-Z2                                  # Board name
export PYNQ_PYTHON=python3.10
export PYNQ_JUPYTER_NOTEBOOKS=/home/xilinx/jupyter_notebooks
export PATH=/usr/local/share/pynq-venv/bin:$PATH
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Source profile.d scripts (additional setup)
for f in /etc/profile.d/*.sh; do
    if [ -r "$f" ]; then
        source "$f" 2>/dev/null || true
    fi
done

# Re-export critical vars (in case profile.d clobbered them)
export XILINX_XRT=/usr
export BOARD=Pynq-Z2

# Run inference (-u for unbuffered output)
cd /home/xilinx/jupyter_notebooks/mobilenet
echo "[run_inference.sh] BOARD=$BOARD, XILINX_XRT=$XILINX_XRT"
/usr/local/share/pynq-venv/bin/python3 -u inference.py
