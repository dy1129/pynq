# run_layer0.py
# First-layer test: CONVBNReLU 0  (224x224x3 -> 112x112x32, k=3, s=2)
#
# Prerequisites (must be set by previous notebook cells):
#   ol           - Overlay
#   image_data   - numpy int32, len 150528 (224*224*3)
#   weights_CONV, bias_CONV, weights_FC, bias_FC  - numpy int32
#   tile_3x3, tile_convs, tile_avg, tile_fc       - numpy int32
#   info_3x3, info_convs, info_avg, info_fc       - numpy int32
#   buf_w_conv, buf_b_conv, buf_w_fc, buf_b_fc    - PYNQ allocate buffers
#   buf_tile_3x3, buf_info_3x3                    - PYNQ allocate buffers
#
# After running, you'll have:
#   buf_input    - packed 3x3 input (sent to IP for each PE)
#   buf_out_pe[0..3] - per-PE output (received from IP)
#   out_layer0   - reassembled 32x112x112 output (CHW int32)

import numpy as np
import time
from pynq import allocate

# ---- Layer 0 parameters ----
L0_DEPTH_OUT = 32
L0_DEPTH_IN  = 3
L0_LENGTH    = 224
L0_STRIDE    = 2
L0_LEN_OUT   = L0_LENGTH // L0_STRIDE   # 112

# ---- Step 1. Pack input (DRAM_2_STREAM_3x3 type=0) ----
print("=== Packing input (this may take 30-60s in pure Python) ===")
t0 = time.time()
packed_in = dram_2_stream_3x3_type0(image_data, L0_DEPTH_OUT, L0_DEPTH_IN, L0_LENGTH)
print(f"  packed input: {len(packed_in):,} ints  ({time.time()-t0:.1f}s)")

# Allocate input DDR buffer and copy
buf_input = allocate(packed_in.shape, dtype=np.int32)
buf_input[:] = packed_in
buf_input.flush()
print(f"  buf_input phys = 0x{buf_input.physical_address:08x}")

# ---- Step 2. Allocate per-PE output buffers ----
# Each PE produces depth/4 channels of 112x112 = 8 * 12544 = 100352 ints
PE_OUT_LEN = (L0_DEPTH_OUT // NUMBER_PE) * L0_LEN_OUT * L0_LEN_OUT
buf_out_pe = []
for pe in range(NUMBER_PE):
    b = allocate((PE_OUT_LEN,), dtype=np.int32)
    buf_out_pe.append(b)
print(f"  per-PE output buffer: {PE_OUT_LEN:,} ints x 4 PE")

# ---- Step 3. DMA channels ----
dma_in  = ol.axi_dma_0.sendchannel
dma_out = ol.axi_dma_1.recvchannel

# ---- Step 4. Call IP for each PE ----
# tile_3x3 / info_3x3 layout:  PE-major, [PE][MAX_CONV_3X3][3]
# offset for PE p (in ints): p * 3 * MAX_CONV_3X3
# offset in bytes:            p * 3 * MAX_CONV_3X3 * 4
TILE_PE_STRIDE_BYTES = 3 * MAX_CONV_3X3 * 4
INFO_PE_STRIDE_BYTES = SIZE_INFO * MAX_CONV_3X3 * 4

print("\n=== Calling IP per PE ===")
print(f"  ap_idle before = {ip_idle()}")

for pe in range(NUMBER_PE):
    tile_phys = buf_tile_3x3.physical_address + pe * TILE_PE_STRIDE_BYTES
    info_phys = buf_info_3x3.physical_address + pe * INFO_PE_STRIDE_BYTES

    print(f"  PE {pe}: tile=0x{tile_phys:08x}, info=0x{info_phys:08x} ...", end=' ')
    try:
        elapsed = ip_call_pe(
            layer=0, inter_layer=0, type_layer=0,
            tile_phys=tile_phys, info_phys=info_phys,
            in_buf=buf_input, out_buf=buf_out_pe[pe],
            dma_in=dma_in, dma_out=dma_out,
            timeout=30.0,
        )
        print(f"done in {elapsed*1000:.1f} ms")
    except TimeoutError as e:
        print(f"TIMEOUT  ({e})")
        print(f"  ap_ctrl now = 0x{ctrl.read(REG_AP_CTRL):08x}")
        break

# ---- Step 5. Reassemble 32x112x112 output ----
print("\n=== Reassembling output ===")
out_layer0 = np.zeros(L0_DEPTH_OUT * L0_LEN_OUT * L0_LEN_OUT, dtype=np.int32)
for pe in range(NUMBER_PE):
    n = stream_2_dram_3x3_pe(out_layer0, np.asarray(buf_out_pe[pe]),
                              pe, L0_DEPTH_OUT, L0_LENGTH, L0_STRIDE)
    print(f"  PE {pe}: wrote {n:,} ints")

# ---- Step 6. Sanity check ----
print("\n=== Output sanity ===")
print(f"  shape: {out_layer0.shape}")
print(f"  min/max: {out_layer0.min()} / {out_layer0.max()}")
print(f"  nonzero ratio: {np.count_nonzero(out_layer0)/len(out_layer0)*100:.1f}%")
print(f"  first 8 values: {out_layer0[:8]}")
