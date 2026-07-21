# Strawberry PYNQ Runtime

PYNQ-Z2 FPGA에서 MobileNet 기반 식물 병해 분류를 실행하기 위한 런타임 파일입니다. 원본 전체 프로젝트 중 `04_jupiter` 폴더만 분리한 배포용 구성입니다.

## Directory Structure

```text
.
├── mobilenet/
│   ├── inference_server.py      # FPGA 추론 백엔드 서버
│   ├── http_server_fpga.py      # 이미지 업로드/결과 조회 HTTP 서버
│   ├── inference.py             # PYNQ Overlay/DMA 기반 추론 코드
│   ├── prepare_image.py         # JPG -> image_int.dat 전처리
│   ├── mobilenet.bit            # FPGA bitstream
│   ├── mobilenet.hwh            # PYNQ hardware handoff
│   ├── weights.dat              # FPGA weight data
│   ├── bias.dat                 # FPGA bias data
│   ├── tile_*.bin               # tile metadata
│   ├── info_*.bin               # layer/info metadata
│   ├── classifier_5cls/         # 1280-dim feature -> 5-class classifier
│   └── python/                  # DMA/stream/helper scripts
└── plants/                      # sample plant images
```

## Target Path on PYNQ

Copy the `mobilenet` folder to:

```bash
/home/xilinx/jupyter_notebooks/mobilenet
```


## PYNQ Runtime Files

PYNQ 보드의 기본 실행 경로는 다음으로 가정합니다.

```bash
/home/xilinx/jupyter_notebooks/mobilenet
```

이 경로에 `04_jupiter/mobilenet` 안의 파일들이 있어야 합니다. 특히 아래 파일은 런타임에 필요합니다.

- `mobilenet.bit`, `mobilenet.hwh`
- `weights.dat`, `bias.dat`
- `tile_3x3.bin`, `tile_convs.bin`, `tile_avg.bin`, `tile_fc.bin`
- `info_3x3.bin`, `info_convs.bin`, `info_avg.bin`, `info_fc.bin`
- `classifier_5cls/*.npy`, `classifier_5cls/*.pkl`
- `inference.py`, `inference_server.py`, `http_server_fpga.py`, `prepare_image.py`

## Run

### 0. SSH

```bash
ssh xilinx@192.168.2.99
ssh xilinx@192.168.0.39
```

### 1. Backend

```bash
cd /home/xilinx/jupyter_notebooks/mobilenet
sudo nohup env XILINX_XRT=/usr BOARD=Pynq-Z2 \
    /usr/local/share/pynq-venv/bin/python3 -u inference_server.py \
    > /tmp/inference_server.log 2>&1 < /dev/null &
```

Debug mode:

```bash
cd /home/xilinx/jupyter_notebooks/mobilenet
sudo env XILINX_XRT=/usr BOARD=Pynq-Z2 \
    /usr/local/share/pynq-venv/bin/python3 inference_server.py
```

### 2. Frontend

Run after the backend is ready.

```bash
cd /home/xilinx/jupyter_notebooks/mobilenet
sudo nohup env XILINX_XRT=/usr BOARD=Pynq-Z2 \
    /usr/local/share/pynq-venv/bin/python3 -u http_server_fpga.py \
    > /tmp/http_server.log 2>&1 < /dev/null &
```

Debug mode:

```bash
cd /home/xilinx/jupyter_notebooks/mobilenet
sudo env XILINX_XRT=/usr BOARD=Pynq-Z2 \
    /usr/local/share/pynq-venv/bin/python3 http_server_fpga.py
```

### 3. Logs

```bash
ssh xilinx@192.168.2.99 "tail -f /tmp/http_server.log"
ssh xilinx@192.168.1.130 "tail -f /tmp/http_server.log"
```

### 4. Status

```bash
ps aux | grep -E 'inference_server|http_server' | grep -v grep
```

Two lines should appear.

### 5. Stop

```bash
sudo pkill -f http_server_fpga
sudo pkill -f inference_server
sudo /usr/local/share/pynq-venv/bin/python3 -c "from pynq import MMIO; MMIO(0x40020000, 0x10000).write(0x00, 0)"
```

## HTTP API

`http_server_fpga.py` runs on port `8080`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload JPEG image |
| `POST` | `/disease` | Upload JPEG image |
| `GET` | `/result` | Get latest result as `class,confidence` |
| `GET` | `/latest.jpg` | View latest uploaded image |
| `GET` | `/` | Simple live view page |

## Output Classes

| Index | Class |
|---:|---|
| 0 | Normal |
| 1 | Strawberry Powdery Mildew |
| 2 | Strawberry Gray Mold |
| 3 | Tomato Leaf Mold |
| 4 | Tomato Yellow Leaf Curl Virus |
```