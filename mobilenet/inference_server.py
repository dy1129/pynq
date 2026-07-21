#!/usr/bin/env python3
"""
PYNQ Inference Server — 무한 루프로 이미지 추론 처리

1회 setup (40초): bitstream load + buffers
매번 추론 (30초): /tmp/infer_request 파일 감지 → 추론 → /tmp/infer_done

Usage on PYNQ:
    sudo XILINX_XRT=/usr BOARD=Pynq-Z2 \\
        /usr/local/share/pynq-venv/bin/python3 -u inference_server.py

Stop: kill -INT <pid>
"""
import os, sys, time, signal, json, traceback

os.chdir('/home/xilinx/jupyter_notebooks/mobilenet')
print("[server] Starting setup...", flush=True)
SETUP_START = time.time()

# === 1회 setup: 모든 cells 1-6 + helpers ===
# inference.py 그대로 사용 (cells 1-18 모두 정의되지만 실제 추론은 함수로 wrapping)
INFERENCE_PATH = '/home/xilinx/jupyter_notebooks/mobilenet/inference.py'

# inference.py를 import 형태로 부분 실행 — cells 1-6만 실행하고 setup
# 단순화: inference.py 전체 실행해서 모든 globals 준비, 마지막에 추론 함수 wrap

# Read inference.py source
with open(INFERENCE_PATH) as f:
    source = f.read()

# ⭐ 옵션 B Step 1: cell 9 (helpers ~470줄) 을 setup으로 이동
# Source 순서: cells 1-6 → cell 7, 8 (image-dep) → cell 9 (helpers) → cell 10-19 (image-dep)
# 재정렬:
#   setup    = cells 1-6  +  cell 9 (helpers — 1회만 정의)
#   inference = cells 7, 8 + cells 10-19 (이미지 의존, 매번 실행)
M_C7  = '# In[7]:'
M_C9  = '# In[9]:'
M_C10 = '# In[10]:'

if all(m in source for m in [M_C7, M_C9, M_C10]):
    idx_c7  = source.index(M_C7)
    idx_c9  = source.index(M_C9)
    idx_c10 = source.index(M_C10)

    setup_part_1_6     = source[:idx_c7]            # cells 1-6
    inference_cells_7_8 = source[idx_c7:idx_c9]      # cells 7, 8 (image-dependent)
    helpers_cell_9     = source[idx_c9:idx_c10]      # cell 9 (helpers — moved to setup)
    inference_cells_10p = source[idx_c10:]           # cells 10-19 (image-dependent)

    setup_src     = setup_part_1_6 + '\n' + helpers_cell_9
    inference_src = inference_cells_7_8 + '\n' + inference_cells_10p
    print(f"[server] Source split: setup={len(setup_src)} chars (with helpers), "
          f"inference={len(inference_src)} chars", flush=True)
else:
    # fallback: 옛날 방식
    if M_C7 in source:
        setup_src, inference_src = source.split(M_C7, 1)
        inference_src = M_C7 + inference_src
    else:
        setup_src = source
        inference_src = ''
    print(f"[server] WARN: 마커 못 찾음, 옛날 분할 사용", flush=True)

print(f"[server] Running setup (cells 1-6 + helpers)...", flush=True)
exec(compile(setup_src, INFERENCE_PATH, 'exec'), globals())

print(f"[server] Setup done in {time.time()-SETUP_START:.1f}s", flush=True)
print(f"[server] Inference code length: {len(inference_src)} chars", flush=True)

# === 무한 루프: trigger 대기 → 추론 → 완료 신호 ===
TRIGGER = '/tmp/infer_request'
DONE = '/tmp/infer_done'
RESULT = '/home/xilinx/jupyter_notebooks/mobilenet/result.json'

# 시작 시 stale signal file 정리
for f in [TRIGGER, DONE]:
    if os.path.exists(f):
        os.remove(f)

print(f"[server] ⭐ Ready! Waiting for {TRIGGER} ...", flush=True)
print(f"[server]   PC쪽에서: ssh xilinx@... 'touch {TRIGGER}' 후 결과 대기", flush=True)

try:
    while True:
        if os.path.exists(TRIGGER):
            os.remove(TRIGGER)
            t0 = time.time()
            print(f"\n[server] === Inference triggered at {time.strftime('%H:%M:%S')} ===", flush=True)
            try:
                # Reload image_int.dat (사진이 새로 업로드됐을 수 있음)
                global image_data
                image_data = load_dat('image_int.dat')[:IMAGE_SIZE]
                print(f"[server] image_int.dat reloaded ({len(image_data)} ints)", flush=True)

                # cpu_map 초기화 (이전 추론 잔여물 클리어)
                buf_cpu_map[:] = 0
                buf_cpu_map[:401408] = out_layer0   # L0 result (사진 dependent)
                # WAIT — out_layer0는 image_data로부터 계산된 거. image 새로 받으면 다시 계산해야!
                # run_layer0.py를 다시 실행
                print(f"[server] Recomputing Layer 0 (depends on new image)...", flush=True)
                exec(compile(open('python/run_layer0.py').read(), 'python/run_layer0.py', 'exec'), globals())
                buf_cpu_map[:] = 0
                buf_cpu_map[:401408] = out_layer0
                buf_cpu_map.flush()

                # ⭐ Soft reset (옵션 A — Overlay 재생성 X)
                # DMA 레지스터는 건드리지 않음 (reset bit 쓰면 PYNQ가 start() 못 함)
                # IP ap_start만 clear + _first_transfer 플래그만 reset → PYNQ가 자동 재시작
                print(f"[server] Soft reset (IP only, DMA untouched)...", flush=True)
                ctrl.write(0x00, 0x00)   # IP ap_start clear
                for _ch in [ip_in, ip_out, res_rd, res_wr]:
                    _ch._first_transfer = True   # 다음 transfer 시 PYNQ가 start() 호출
                print(f"[server] Soft reset done", flush=True)

                # ⭐ NEW IMAGE 모드: testbench dump 사용 안 함, FPGA 결과만 사용
                globals()['NEW_IMAGE_MODE'] = True
                # 추론 실행 (cells 7-18)
                # 참고: cell 9에서 G3_OUTPUTS = {}로 자동 재초기화됨
                # 단, NEW_IMAGE_MODE는 cell 9에서 False로 다시 set되므로 한번 더 set 필요
                # → 해결: cell 9 안의 NEW_IMAGE_MODE 라인을 패치 (또는 inference_src 수정)
                inference_src_patched = inference_src.replace(
                    'NEW_IMAGE_MODE = False', 'NEW_IMAGE_MODE = True   # patched by server')
                exec(compile(inference_src_patched, INFERENCE_PATH, 'exec'), globals())

                elapsed = time.time() - t0
                print(f"[server] === Done in {elapsed:.1f}s ===", flush=True)
            except Exception as e:
                print(f"[server] ERROR: {e}", flush=True)
                traceback.print_exc()
                # Write error to result.json
                with open(RESULT, 'w', encoding='utf-8') as f:
                    json.dump({'error': str(e)}, f, ensure_ascii=False, indent=2)

            # 완료 신호
            open(DONE, 'w').close()
            print(f"[server] Signal {DONE} written", flush=True)
            print(f"[server] ⭐ Ready for next image ...", flush=True)
        else:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[server] Shutting down...", flush=True)
    sys.exit(0)
