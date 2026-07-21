# helpers.py
# MobileNetV2 PYNQ-Z2 inference helpers
# Ports DRAM_2_STREAM_3x3 (type=0) + STREAM_2_DRAM_3x3 + IP control from
# memory_access.cpp / MobileNet_TB.cpp.
#
# Usage from a Jupyter notebook:
#   %run helpers.py
# After running, the following are available:
#   ctrl, data           - MMIO objects for IP CTRL_BUS and DATA_BUS
#   dram_2_stream_3x3_type0(in_mem, depth_out, depth_in, length)
#   stream_2_dram_3x3_pe(out_mem, pe_data, pe_idx, depth, length, stride)
#   ip_call_pe(...)
#   etc.

import numpy as np
import time
from pynq import MMIO

# ---- Constants from parameters.h / layer.h ----
TILE_MAP        = 28
TILE_CONV_OUT   = 32
NUMBER_PE       = 4
SIZE_INFO       = 19
MAX_CONV_3X3    = 64
MAX_CONVS       = 100
MAX_AVG         = 10
MAX_FC          = 160

# ---- IP register map (from xmobilenet_stream_hw.h) ----
IP_CTRL_BASE = 0x40000000   # CTRL_BUS  (ap_ctrl, layer, inter_layer, type_layer)
IP_DATA_BASE = 0x40010000   # control bus (pointers ext_w_conv ... ext_info)

# CTRL_BUS offsets
REG_AP_CTRL     = 0x00
REG_LAYER       = 0x10
REG_INTER_LAYER = 0x18
REG_TYPE_LAYER  = 0x20

# DATA bus offsets (lo/hi for 64-bit ptrs)
REG_W_CONV = 0x10
REG_B_CONV = 0x1C
REG_W_FC   = 0x28
REG_B_FC   = 0x34
REG_TILE   = 0x40
REG_INFO   = 0x4C

# ---- Construct MMIO once ----
ctrl = MMIO(IP_CTRL_BASE, 0x10000)
data = MMIO(IP_DATA_BASE, 0x10000)

# ---- Pointer / arg helpers ----
def write_ptr64(mmio, off, addr):
    mmio.write(off,     addr & 0xFFFFFFFF)
    mmio.write(off + 4, (addr >> 32) & 0xFFFFFFFF)

def ip_set_args(layer, inter_layer, type_layer):
    ctrl.write(REG_LAYER,       int(layer))
    ctrl.write(REG_INTER_LAYER, int(inter_layer))
    ctrl.write(REG_TYPE_LAYER,  int(type_layer))

def ip_set_tile_info(tile_addr, info_addr):
    write_ptr64(data, REG_TILE, tile_addr)
    write_ptr64(data, REG_INFO, info_addr)

def ip_start():
    ctrl.write(REG_AP_CTRL, 1)

def ip_wait(timeout=20.0):
    t0 = time.time()
    while True:
        v = ctrl.read(REG_AP_CTRL)
        if (v >> 1) & 1:        # ap_done
            return time.time() - t0
        if time.time() - t0 > timeout:
            raise TimeoutError(f"IP timeout, ap_ctrl=0x{v:08x}")
        time.sleep(0.0001)

def ip_idle():
    return (ctrl.read(REG_AP_CTRL) >> 2) & 1

# ---- DRAM_2_STREAM_3x3 type_layer=0 (full 3x3 conv with padding) ----
# Identical sequence as memory_access.cpp lines 142-206.
# In testbench it broadcasts to 4 PE streams; we generate 1 buffer and
# resend it via DMA for each PE call.
def dram_2_stream_3x3_type0(in_mem, depth_out, depth_in, length):
    multi_max_in = length * length
    limit_map    = length - 1
    out = []
    push = out.append
    for j in range(0, depth_in, TILE_CONV_OUT):
        limit_j = min(depth_in, j + TILE_CONV_OUT)
        for k in range(0, length, TILE_MAP):
            limit_k = min(length, k + TILE_MAP) + 1
            min_k   = k - 1
            size_x  = limit_k - k + 1
            for l in range(0, length, TILE_MAP):
                limit_l = min(length, l + TILE_MAP) + 1
                min_l   = l - 1
                for _i in range(min(depth_out, TILE_CONV_OUT) // NUMBER_PE):
                    for m in range(j, limit_j):
                        pos_x = m * multi_max_in
                        for n in range(min_k, limit_k):
                            if 0 <= n <= limit_map:
                                pos_y = n * length + pos_x
                                for o in range(min_l, limit_l):
                                    if 0 <= o <= limit_map:
                                        push(int(in_mem[o + pos_y]))
                                    else:
                                        push(0)
                            else:
                                for _ in range(size_x):
                                    push(0)
    return np.array(out, dtype=np.int32)

# ---- STREAM_2_DRAM_3x3 expansion=1 (per-PE write back) ----
# Each PE writes its slice (depth/4 channels) into out_mem at correct CHW positions.
def stream_2_dram_3x3_pe(out_mem, pe_data, pe_idx, depth, length, stride):
    length_out   = length // stride
    multi_max_in = length_out * length_out
    cnt = 0
    for i in range(pe_idx * depth // NUMBER_PE, (pe_idx + 1) * depth // NUMBER_PE, TILE_CONV_OUT):
        min_i = min((pe_idx + 1) * depth // NUMBER_PE, i + TILE_CONV_OUT)
        for j in range(0, length, TILE_MAP):
            min_j = min(length, j + TILE_MAP)
            for k in range(0, length, TILE_MAP):
                min_k = min(length, k + TILE_MAP)
                for l in range(i, min_i):
                    pos_x = l * multi_max_in
                    for m in range(j // stride, min_j // stride):
                        pos_y = m * length_out + pos_x
                        for o in range(k // stride, min_k // stride):
                            if cnt < len(pe_data):
                                out_mem[o + pos_y] = pe_data[cnt]
                            cnt += 1
    return cnt

# ---- High-level: call IP for one PE ----
# Caller must have set ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc once already.
# This sets per-call ext_tile/ext_info, programs DMAs, and waits for ap_done.
#
#   tile_phys, info_phys: physical addresses (with PE offset already added)
#   in_buf, out_buf: PYNQ allocate() buffers (.physical_address used by DMA)
#   layer, inter_layer, type_layer: scalar args to IP
#   dma_in: ol.axi_dma_0.sendchannel
#   dma_out: ol.axi_dma_1.recvchannel
def ip_call_pe(layer, inter_layer, type_layer,
               tile_phys, info_phys,
               in_buf, out_buf, dma_in, dma_out, timeout=20.0):
    ip_set_args(layer, inter_layer, type_layer)
    ip_set_tile_info(tile_phys, info_phys)
    # Start DMA RX first (must be ready before IP starts producing)
    dma_out.transfer(out_buf)
    dma_in.transfer(in_buf)
    ip_start()
    elapsed = ip_wait(timeout=timeout)
    dma_in.wait()
    dma_out.wait()
    return elapsed

print("helpers.py loaded.")
print(f"  ctrl @ 0x{IP_CTRL_BASE:08x}, data @ 0x{IP_DATA_BASE:08x}")
print(f"  ap_idle = {ip_idle()}")
