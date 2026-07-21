# run_layer1.py
# InvertedResidual 1 (special: 2 groups, no residual)
#   Group 1 (depthwise 3x3): layer=1, inter=0, type=2, in=32, out=32, length=112
#   Group 2 (1x1 conv):      layer=1, inter=1, type=1, in=32, out=16, length=112
#
# Prerequisites:
#   - %run -i helpers.py    (provides ctrl, data, ip_*, MMIO, basic constants)
#   - %run -i helpers_v2.py (provides dram_2_stream_*, conv_batch_relu_pe, etc.)
#   - %run -i run_layer0.py (provides out_layer0)
#   - buf_w_conv, buf_b_conv, buf_tile_convs, buf_info_convs   (already allocated)
#   - ip_in, ip_out, res_rd, res_wr                            (DMA channels)

import numpy as np
import time
from pynq import allocate

# ---- Layer 1 parameters ----
L1G1_DEPTH    = 32        # depthwise: in=out=32
L1G1_LENGTH   = 112
L1G1_STRIDE   = 1

L1G2_DEPTH_OUT = 16
L1G2_DEPTH_IN  = 32
L1G2_LENGTH    = 112
L1G2_STRIDE    = 1

# ============================================================
# Step 0. Allocate cpu_map (DDR feature map buffer)
# ============================================================
print("=== Allocating cpu_map ===")
CPU_MAP_SIZE = 1572864    # parameters.h: map_size
buf_cpu_map = allocate((CPU_MAP_SIZE,), dtype=np.int32)
print(f"  buf_cpu_map phys = 0x{buf_cpu_map.physical_address:08x}, "
      f"{buf_cpu_map.nbytes:,} bytes")

# Copy layer 0 output (CHW format from STREAM_2_DRAM_3x3 expansion=1)
buf_cpu_map[:401408] = out_layer0
buf_cpu_map.flush()
print(f"  cpu_map filled with layer 0 output (32x112x112 = 401408 ints)")
print(f"  cpu_map[:8] = {np.asarray(buf_cpu_map)[:8]}")

cpu_map_np = np.asarray(buf_cpu_map)


# ============================================================
# Step 1. Group 1 — Depthwise 3x3
# ============================================================
print("\n=== Layer 1 Group 1: depthwise 3x3 (in=32, out=32, k=3, s=1) ===")

buf_l1g1_outputs = []
PE_OUT_LEN_G1 = (L1G1_DEPTH // NUMBER_PE) * L1G1_LENGTH * L1G1_LENGTH   # 8*112*112=100352

for pe in range(NUMBER_PE):
    t0 = time.time()
    packed_in = dram_2_stream_3x3_type2(cpu_map_np, L1G1_DEPTH, L1G1_LENGTH, pe)
    print(f"  PE {pe}: packed input {len(packed_in):,} ints in {time.time()-t0:.1f}s")

    buf_in = allocate(packed_in.shape, dtype=np.int32)
    buf_in[:] = packed_in
    buf_in.flush()

    buf_out = allocate((PE_OUT_LEN_G1,), dtype=np.int32)

    elapsed = conv_batch_relu_pe(
        layer=1, inter_layer=0, type_layer=2, pe=pe,
        in_buf=buf_in, out_buf=buf_out,
        dma_in=ip_in, dma_out=ip_out, timeout=30.0,
    )
    print(f"    IP done in {elapsed*1000:.1f} ms")

    buf_l1g1_outputs.append(buf_out)
    del buf_in   # free DDR

# Write Group 1 result back to cpu_map (expansion=0, sequential)
print("\n  Writing Group 1 output to cpu_map (sequential, expansion=0)")
n_written = stream_2_dram_3x3_exp0(buf_cpu_map, buf_l1g1_outputs,
                                    L1G1_DEPTH, L1G1_LENGTH, L1G1_STRIDE)
buf_cpu_map.flush()
print(f"  Wrote {n_written:,} ints to cpu_map")

# Sanity check Group 1 output
g1_total = L1G1_DEPTH * L1G1_LENGTH * L1G1_LENGTH
g1_view = cpu_map_np[:g1_total]
print(f"  Group 1 stats: nonzero {np.count_nonzero(g1_view)/len(g1_view)*100:.1f}%, "
      f"min={g1_view.min()}, max={g1_view.max()}")
print(f"  Group 1 first 16: {g1_view[:16]}")


# ============================================================
# Step 2. Group 2 — 1x1 conv (32 -> 16)
# ============================================================
print("\n=== Layer 1 Group 2: 1x1 conv (in=32, out=16, k=1, s=1) ===")

buf_l1g2_outputs = []
PE_OUT_LEN_G2 = (L1G2_DEPTH_OUT // NUMBER_PE) * L1G2_LENGTH * L1G2_LENGTH  # 4*112*112=50176

for pe in range(NUMBER_PE):
    t0 = time.time()
    packed_in = dram_2_stream_1x1_exp0(cpu_map_np, L1G2_DEPTH_OUT, L1G2_DEPTH_IN,
                                        L1G2_LENGTH, L1G2_STRIDE, pe)
    print(f"  PE {pe}: packed input {len(packed_in):,} ints in {time.time()-t0:.1f}s")

    buf_in = allocate(packed_in.shape, dtype=np.int32)
    buf_in[:] = packed_in
    buf_in.flush()

    buf_out = allocate((PE_OUT_LEN_G2,), dtype=np.int32)

    elapsed = conv_batch_relu_pe(
        layer=1, inter_layer=1, type_layer=1, pe=pe,
        in_buf=buf_in, out_buf=buf_out,
        dma_in=ip_in, dma_out=ip_out, timeout=30.0,
    )
    print(f"    IP done in {elapsed*1000:.1f} ms")

    buf_l1g2_outputs.append(buf_out)
    del buf_in

# Write Group 2 result back to cpu_map (expansion=0)
print("\n  Writing Group 2 output to cpu_map (sequential, expansion=0)")
n_written = stream_2_dram_1x1_exp0(buf_cpu_map, buf_l1g2_outputs,
                                    L1G2_DEPTH_OUT, L1G2_LENGTH)
buf_cpu_map.flush()
print(f"  Wrote {n_written:,} ints to cpu_map")

# Final sanity check
g2_total = L1G2_DEPTH_OUT * L1G2_LENGTH * L1G2_LENGTH
g2_view = cpu_map_np[:g2_total]
print(f"\n=== Layer 1 Output Sanity ===")
print(f"  Shape (1D):          {g2_view.shape}  (= 16*112*112 = {g2_total})")
print(f"  Range:               [{g2_view.min()}, {g2_view.max()}]")
print(f"  Nonzero ratio:       {np.count_nonzero(g2_view)/len(g2_view)*100:.1f}%")
print(f"  First 16:            {g2_view[:16]}")
print(f"  Channel 0 [0:8]:     {g2_view[0:8]}")
print(f"  Channel 5 [0:8]:     {g2_view[5*12544:5*12544+8]}")
print(f"  Channel 15 [0:8]:    {g2_view[15*12544:15*12544+8]}")
print(f"\nFirst nonzero positions: {np.nonzero(g2_view)[0][:5]}")
