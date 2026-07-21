# helpers_v3.py
# Authoritative port of Hardware/Vitis/memory_access.c (baremetal SDK).
# Vitis SDK is the source of truth: it is the working code on a real Zynq board.
#
# Usage from Jupyter:
#   %run -i helpers.py
#   %run -i helpers_v2.py    (still useful for some functions)
#   %run -i helpers_v3.py    (overrides with SDK-correct versions)

import numpy as np
import time
from pynq import allocate

# ---- Constants (parameters.h) ----
TILE_MAP        = 28
TILE_CONV_OUT   = 32
TILE_CONV_IN    = 32
NUMBER_PE       = 4
SIZE_INFO       = 19
MAX_CONV_3X3    = 64
MAX_CONVS       = 100
MAX_AVG         = 10
MAX_FC          = 160
TILE_FC_OUT     = 8     # parameters.h: tile_fc_out
TILE_FC_IN      = 1280  # parameters.h: tile_fc_in (estimate)


# ============================================================
# DRAM_2_STREAM_3x3 (SDK lines 54-152)
# Identical to testbench/v2 for type 0/2.
# ============================================================
def dram_2_stream_3x3_v3(in_mem, depth_out, depth_in, length, type_layer):
    """SDK port. Returns one buffer that contains data for ALL 4 PEs (concatenated).
       For type=0: same data 4x (broadcast).
       For type=2: PE-specific data 4x (different segments).
       In PYNQ we'll split this into 4 buffers and feed each PE."""
    multi_max_in = length * length
    limit_map = length - 1
    depth_in_PE = depth_in // NUMBER_PE
    out = []
    push = out.append

    if type_layer == 0:
        # 3x3 standard - broadcast same data
        # SDK builds 1 stream; in PYNQ same buffer is sent to each PE call.
        for j in range(0, depth_in, TILE_CONV_OUT):
            limit_j = min(depth_in, j + TILE_CONV_OUT)
            for k in range(0, length, TILE_MAP):
                limit_k = min(length, k + TILE_MAP) + 1
                min_k = k - 1
                size_x = limit_k - k + 1
                for l in range(0, length, TILE_MAP):
                    limit_l = min(length, l + TILE_MAP) + 1
                    min_l = l - 1
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
        per_pe = len(out)
        return np.array(out, dtype=np.int32), per_pe

    else:  # type 2 - depthwise per PE
        per_pe_lengths = []
        for PE in range(NUMBER_PE):
            pe_start_len = len(out)
            for j in range(PE * depth_in_PE, (PE + 1) * depth_in_PE, TILE_CONV_OUT):
                limit_j = min((PE + 1) * depth_in_PE, j + TILE_CONV_OUT)
                for k in range(0, length, TILE_MAP):
                    limit_k = min(length, k + TILE_MAP) + 1
                    min_k = k - 1
                    size_x = limit_k - k + 1
                    for l in range(0, length, TILE_MAP):
                        limit_l = min(length, l + TILE_MAP) + 1
                        min_l = l - 1
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
            per_pe_lengths.append(len(out) - pe_start_len)
        return np.array(out, dtype=np.int32), per_pe_lengths[0]


# ============================================================
# DRAM_2_STREAM_1x1 (SDK lines 154-251)
# stride=1: simple broadcast (lines 167-190)
# stride>1: per-PE strided memcpy (lines 191-218)
# type=3: AVG layer (lines 220-248)
# ============================================================
def dram_2_stream_1x1_v3(in_mem, depth_out, depth_in, length, type_layer, stride):
    """Returns concatenated buffer. For PYNQ, send same buffer to each PE call."""
    multi_max_in = length * length
    data_for_PE = length * length * depth_in // NUMBER_PE
    depth_out_PE = depth_out // NUMBER_PE
    depth_in_PE  = depth_in // NUMBER_PE
    out = []
    push = out.append

    if type_layer == 1:
        if stride == 1:
            # SDK lines 167-190: simple broadcast
            for k in range(0, length, TILE_MAP):
                min_k = min(length, k + TILE_MAP)
                for l in range(0, length, TILE_MAP):
                    min_l = min(length, l + TILE_MAP)
                    for i in range(0, depth_out_PE, TILE_CONV_OUT):
                        for j in range(0, depth_in, TILE_CONV_IN):
                            min_j = min(depth_in, j + TILE_CONV_IN)
                            for m in range(j, min_j):
                                pos_x = m * multi_max_in
                                for n in range(k, min_k):
                                    pos_y = n * length + pos_x
                                    for o in range(l, min_l):
                                        push(int(in_mem[o + pos_y]))
        else:
            # SDK lines 191-218: stride>1, PE-specific memcpy
            in_arr = np.asarray(in_mem)
            k_pos = 0
            for k in range(0, length, TILE_MAP):
                min_k = min(length, k + TILE_MAP)
                limit_k = min_k - k
                l_pos = 0
                for l in range(0, length, TILE_MAP):
                    min_l = min(length, l + TILE_MAP)
                    limit_l = min_l - l
                    add_lk = l_pos + k_pos
                    mul_lk = limit_l * limit_k
                    for i in range(0, depth_out_PE, TILE_CONV_OUT):
                        for PE in range(NUMBER_PE):
                            aux = PE * data_for_PE
                            j_pos = 0
                            for j in range(PE * depth_in_PE, (PE + 1) * depth_in_PE, TILE_CONV_IN):
                                min_j = min((PE + 1) * depth_in_PE, j + TILE_CONV_IN)
                                limit_j = min_j - j
                                counter_in = add_lk * limit_j + j_pos + aux
                                n_xfer = mul_lk * limit_j
                                for x in range(counter_in, counter_in + n_xfer):
                                    push(int(in_arr[x]))
                                j_pos += multi_max_in * limit_j
                    l_pos += mul_lk
                k_pos += length * limit_k

    elif type_layer == 3:
        # SDK lines 220-248: AVG, per-PE
        for PE in range(NUMBER_PE):
            for k in range(0, length, TILE_MAP):
                min_k = min(length, k + TILE_MAP)
                for l in range(0, length, TILE_MAP):
                    min_l = min(length, l + TILE_MAP)
                    for j in range(PE * depth_in_PE, (PE + 1) * depth_in_PE, TILE_CONV_IN):
                        min_j = min((PE + 1) * depth_in_PE, j + TILE_CONV_IN)
                        for m in range(j, min_j):
                            pos_x = m * multi_max_in
                            for n in range(k, min_k):
                                pos_y = n * length + pos_x
                                for o in range(l, min_l):
                                    push(int(in_mem[o + pos_y]))

    return np.array(out, dtype=np.int32)


# ============================================================
# DRAM_2_STREAM_array (SDK lines 267-276) — for FC layer input
# ============================================================
def dram_2_stream_array_v3(in_mem, length_out, length_in):
    """For FC: replicate input length_in times for each FC output tile.
       Returns concatenated buffer (will be sent broadcast to each PE)."""
    in_arr = np.asarray(in_mem)
    n_repeat = length_out // NUMBER_PE // TILE_FC_OUT
    if n_repeat < 1: n_repeat = 1
    return np.tile(in_arr[:length_in], n_repeat).astype(np.int32)


# ============================================================
# STREAM_2_DRAM_3x3 (SDK lines 278-310)
# Reorders raw IP outputs into CHW layout in cpu_map.
# Each PE's output is in_mem[counter++] sequence.
# ============================================================
def stream_2_dram_3x3_v3(out_mem, pe_data_list, depth, length, stride):
    """SDK port. pe_data_list is list of 4 numpy arrays (one per PE).
       Writes to out_mem in CHW layout (channel*length_out^2 + row*length_out + col)."""
    length_out = length // stride
    multi_max_in = length_out * length_out
    depth_PE = depth // NUMBER_PE
    out_np = np.asarray(out_mem)

    for PE in range(NUMBER_PE):
        pe_data = np.asarray(pe_data_list[PE])
        counter = 0
        for i in range(PE * depth_PE, (PE + 1) * depth_PE, TILE_CONV_OUT):
            min_i = min((PE + 1) * depth_PE, i + TILE_CONV_OUT)
            for j in range(0, length, TILE_MAP):
                min_j = min(length, j + TILE_MAP) // stride
                j_stride = j // stride
                for k in range(0, length, TILE_MAP):
                    min_k = min(length, k + TILE_MAP) // stride
                    k_stride = k // stride
                    for l in range(i, min_i):
                        pos_x = l * multi_max_in
                        for m in range(j_stride, min_j):
                            pos_y = m * length_out + pos_x
                            for o in range(k_stride, min_k):
                                if counter < len(pe_data):
                                    out_np[o + pos_y] = pe_data[counter]
                                counter += 1


# ============================================================
# STREAM_2_DRAM_1x1 (SDK lines 312-344)
# ============================================================
def stream_2_dram_1x1_v3(out_mem, pe_data_list, depth, length):
    multi_max_in = length * length
    depth_PE = depth // NUMBER_PE
    out_np = np.asarray(out_mem)

    for PE in range(NUMBER_PE):
        pe_data = np.asarray(pe_data_list[PE])
        counter = 0
        for j in range(0, length, TILE_MAP):
            min_j = min(length, j + TILE_MAP)
            for k in range(0, length, TILE_MAP):
                min_k = min(length, k + TILE_MAP)
                for i in range(PE * depth_PE, (PE + 1) * depth_PE, TILE_CONV_OUT):
                    min_i = min((PE + 1) * depth_PE, i + TILE_CONV_OUT)
                    for l in range(i, min_i):
                        pos_x = l * multi_max_in
                        for m in range(j, min_j):
                            pos_y = m * length + pos_x
                            for o in range(k, min_k):
                                if counter < len(pe_data):
                                    out_np[o + pos_y] = pe_data[counter]
                                counter += 1


# ============================================================
# STREAM_2_DRAM_array (for FC output)
# ============================================================
def stream_2_dram_array_v3(out_mem, pe_data_list, length):
    """Concatenate 4 PE arrays into out_mem."""
    length_PE = length // NUMBER_PE
    out_np = np.asarray(out_mem)
    for PE in range(NUMBER_PE):
        d = np.asarray(pe_data_list[PE])
        out_np[PE * length_PE : PE * length_PE + len(d)] = d


# ============================================================
# Convenience: run one PE of a layer
# Same as helpers_v2.conv_batch_relu_pe but parameterized cleaner
# ============================================================
def call_ip_pe(layer, inter_layer, type_layer, pe,
               in_buf, out_buf, dma_in, dma_out, timeout=20.0):
    if type_layer == 0:
        tile_phys = buf_tile_3x3.physical_address + tile_3x3_offset_bytes(pe)
        info_phys = buf_info_3x3.physical_address + info_3x3_offset_bytes(pe)
    elif type_layer == 3:
        tile_phys = buf_tile_avg.physical_address + tile_avg_offset_bytes(pe)
        info_phys = buf_info_avg.physical_address + info_avg_offset_bytes(pe)
    elif type_layer == 4:
        tile_phys = buf_tile_fc.physical_address + tile_fc_offset_bytes(pe)
        info_phys = buf_info_fc.physical_address + info_fc_offset_bytes(pe)
    else:  # 1 or 2
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


# Tile/info offsets for AVG and FC (helpers_v2 only had 3x3 and convs)
def tile_avg_offset_bytes(pe):
    return pe * 3 * MAX_AVG * 4

def info_avg_offset_bytes(pe):
    return pe * SIZE_INFO * MAX_AVG * 4

def tile_fc_offset_bytes(pe):
    return pe * 3 * MAX_FC * 4

def info_fc_offset_bytes(pe):
    return pe * SIZE_INFO * MAX_FC * 4


print("helpers_v3.py loaded (SDK-correct port).")
print("  dram_2_stream_3x3_v3, dram_2_stream_1x1_v3, dram_2_stream_array_v3")
print("  stream_2_dram_3x3_v3, stream_2_dram_1x1_v3, stream_2_dram_array_v3")
print("  call_ip_pe (universal IP call helper)")
