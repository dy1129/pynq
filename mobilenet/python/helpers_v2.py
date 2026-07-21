# helpers_v2.py
# Phase 4 helpers for InvertedResidual layers (depthwise 3x3 + 1x1 pointwise)
# Ports DRAM_2_STREAM_3x3(type=2) / DRAM_2_STREAM_1x1(exp=0) / STREAM_2_DRAM_*(exp=0)
# from memory_access.cpp
#
# Usage: %run -i helpers_v2.py   (after %run -i helpers.py)

import numpy as np
import time
from pynq import allocate

# ---- Constants (from parameters.h / layer.h) ----
TILE_MAP        = 28
TILE_CONV_OUT   = 32
TILE_CONV_IN    = 32     # parameters.h
NUMBER_PE       = 4
SIZE_INFO       = 19
MAX_CONV_3X3    = 64
MAX_CONVS       = 100
MAX_LAYER_CONVS = 18     # (layer-1) range: 0..17 for InvertedResidual layers

# ============================================================
# DRAM_2_STREAM_3x3 type_layer=2  (depthwise 3x3, per-PE)
# Each PE handles depth/4 channels, padded sliding window
# Returns 1D int32 array for one PE call
# ============================================================
def dram_2_stream_3x3_type2(in_mem, depth_in, length, pe_idx):
    multi_max_in = length * length
    limit_map = length - 1
    out = []
    push = out.append

    pe_start = pe_idx * depth_in // NUMBER_PE
    pe_end   = (pe_idx + 1) * depth_in // NUMBER_PE

    for j in range(pe_start, pe_end, TILE_CONV_OUT):
        limit_j = min(pe_end, j + TILE_CONV_OUT)
        for k in range(0, length, TILE_MAP):
            limit_k = min(length, k + TILE_MAP) + 1
            min_k   = k - 1
            size_x  = limit_k - k + 1
            for l in range(0, length, TILE_MAP):
                limit_l = min(length, l + TILE_MAP) + 1
                min_l   = l - 1
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


# ============================================================
# DRAM_2_STREAM_1x1 type_layer=1, expansion=0  (per-PE pre-reordered read)
# in_mem layout: PE-major (PE*length*length*depth_in/4 chunks of sequential data)
# This matches what STREAM_2_DRAM_3x3(expansion=0) wrote in the previous step.
# ============================================================
def dram_2_stream_1x1_exp0(in_mem, depth_out, depth_in, length, stride, pe_idx):
    out = []
    push = out.append

    k_pos = 0
    for k in range(0, length, TILE_MAP):
        min_k   = min(length, k + TILE_MAP)
        limit_k = min_k - k
        l_pos = 0
        for l in range(0, length, TILE_MAP):
            min_l   = min(length, l + TILE_MAP)
            limit_l = min_l - l
            for i in range(0, depth_out // NUMBER_PE, TILE_CONV_OUT):
                pe_start = pe_idx * depth_in // NUMBER_PE
                pe_end   = (pe_idx + 1) * depth_in // NUMBER_PE
                j_pos = 0
                for j in range(pe_start, pe_end, TILE_CONV_IN):
                    min_j   = min(pe_end, j + TILE_CONV_IN)
                    limit_j = min_j - j
                    reg = (l_pos * limit_j + k_pos * limit_j + j_pos +
                           pe_idx * (length // stride) * (length // stride) * depth_in // NUMBER_PE)
                    n_xfer = limit_l * limit_k * limit_j // (stride * stride)
                    for x in range(reg, reg + n_xfer):
                        push(int(in_mem[x]))
                    j_pos += (length // stride) * (length // stride) * limit_j
            l_pos += limit_l * limit_k // (stride * stride)
        k_pos += (length // stride) * limit_k // stride
    return np.array(out, dtype=np.int32)


# ============================================================
# DRAM_2_STREAM_1x1 type_layer=1, expansion=1  (broadcast, same data to all 4 PE)
# Used when previous step wrote with expansion=1 layout (CHW)
# ============================================================
def dram_2_stream_1x1_exp1(in_mem, depth_out, depth_in, length):
    multi_max_in = length * length
    out = []
    push = out.append

    for k in range(0, length, TILE_MAP):
        min_k = min(length, k + TILE_MAP)
        for l in range(0, length, TILE_MAP):
            min_l = min(length, l + TILE_MAP)
            for i in range(0, depth_out // NUMBER_PE, TILE_CONV_OUT):
                for j in range(0, depth_in, TILE_CONV_IN):
                    min_j = min(depth_in, j + TILE_CONV_IN)
                    for m in range(j, min_j):
                        pos_x = m * multi_max_in
                        for n in range(k, min_k):
                            pos_y = n * length + pos_x
                            for o in range(l, min_l):
                                push(int(in_mem[o + pos_y]))
    return np.array(out, dtype=np.int32)


# ============================================================
# STREAM_2_DRAM_3x3 expansion=0  (4 PE concatenated sequentially)
# All 4 PEs feed into a single virtual stream; data is written sequentially.
# In PYNQ we call IP 4 times and concat results.
# ============================================================
def stream_2_dram_3x3_exp0(out_mem, pe_data_list, depth, length, stride):
    total = depth * length * length // (stride * stride)
    out_np = np.asarray(out_mem)
    cnt = 0
    for pe_idx in range(NUMBER_PE):
        d = np.asarray(pe_data_list[pe_idx])
        n = len(d)
        room = total - cnt
        if room <= 0:
            break
        n_use = min(n, room)
        out_np[cnt:cnt + n_use] = d[:n_use]
        cnt += n
    return cnt


# ============================================================
# STREAM_2_DRAM_3x3 expansion=1  (per-PE CHW write-back, like layer 0)
# Already in helpers.py as stream_2_dram_3x3_pe — re-export wrapper
# ============================================================
def stream_2_dram_3x3_exp1(out_mem, pe_data_list, depth, length, stride):
    cnt_total = 0
    for pe_idx in range(NUMBER_PE):
        cnt_total += stream_2_dram_3x3_pe(
            np.asarray(out_mem), np.asarray(pe_data_list[pe_idx]),
            pe_idx, depth, length, stride)
    return cnt_total


# ============================================================
# STREAM_2_DRAM_1x1 expansion=0  (4 PE concatenated sequentially)
# ============================================================
def stream_2_dram_1x1_exp0(out_mem, pe_data_list, depth, length):
    total = depth * length * length
    out_np = np.asarray(out_mem)
    cnt = 0
    for pe_idx in range(NUMBER_PE):
        d = np.asarray(pe_data_list[pe_idx])
        n = len(d)
        room = total - cnt
        if room <= 0:
            break
        n_use = min(n, room)
        out_np[cnt:cnt + n_use] = d[:n_use]
        cnt += n
    return cnt


# ============================================================
# Tile / Info offset helpers (in BYTES, for adding to physical_address)
# ============================================================
# tile_3x3 layout (set_tile_info): tile_3x3[PE*3*MAX_CONV_3X3 + i*3 + j]
def tile_3x3_offset_bytes(pe):
    return pe * 3 * MAX_CONV_3X3 * 4

def info_3x3_offset_bytes(pe):
    return pe * SIZE_INFO * MAX_CONV_3X3 * 4

# tile_convs layout: [PE*3*MAX_CONVS*3*18 + i*3*MAX_CONVS*3 + j*3*MAX_CONVS + k*3 + l]
# where i = (layer-1), j = inter_layer, k = conv_block, l = tile_idx
# Per CONV_BATCH_RELU (testbench), for IP call:
#   base = (layer-1)*3*MAX_CONVS*3 + inter_layer*3*MAX_CONVS + PE*3*MAX_CONVS*3*18
def tile_convs_offset_bytes(layer, inter_layer, pe):
    base_ints = (pe * 3 * MAX_CONVS * 3 * 18 +
                 (layer - 1) * 3 * MAX_CONVS * 3 +
                 inter_layer * 3 * MAX_CONVS)
    return base_ints * 4

def info_convs_offset_bytes(layer, inter_layer, pe):
    base_ints = (pe * SIZE_INFO * MAX_CONVS * 3 * 18 +
                 (layer - 1) * SIZE_INFO * MAX_CONVS * 3 +
                 inter_layer * SIZE_INFO * MAX_CONVS)
    return base_ints * 4


# ============================================================
# Convenience: high-level CONV_BATCH_RELU equivalent
# Calls IP for one PE, with correct tile/info pointer offset.
# ============================================================
def conv_batch_relu_pe(layer, inter_layer, type_layer, pe,
                       in_buf, out_buf, dma_in, dma_out, timeout=30.0):
    if type_layer == 0:        # layer-0 style (3x3 standalone)
        tile_phys = buf_tile_3x3.physical_address + tile_3x3_offset_bytes(pe)
        info_phys = buf_info_3x3.physical_address + info_3x3_offset_bytes(pe)
    else:                       # type 1 (1x1) or 2 (depthwise) → tile_convs
        tile_phys = buf_tile_convs.physical_address + tile_convs_offset_bytes(layer, inter_layer, pe)
        info_phys = buf_info_convs.physical_address + info_convs_offset_bytes(layer, inter_layer, pe)

    ip_set_args(layer, inter_layer, type_layer)
    ip_set_tile_info(tile_phys, info_phys)
    dma_out.transfer(out_buf)
    dma_in.transfer(in_buf)
    ip_start()
    elapsed = ip_wait(timeout=timeout)
    dma_in.wait()
    dma_out.wait()
    return elapsed


print("helpers_v2.py loaded.")
print(f"  Functions: dram_2_stream_3x3_type2, dram_2_stream_1x1_exp0/1,")
print(f"             stream_2_dram_3x3_exp0/1, stream_2_dram_1x1_exp0,")
print(f"             tile/info offset helpers, conv_batch_relu_pe")
