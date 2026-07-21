import io, os, time, json, subprocess, threading, base64
import urllib.request
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from pynq import MMIO

SERVER_PORT = 8080
MODEL_DIR = '/home/xilinx/jupyter_notebooks/mobilenet'
PREPARE_PY = os.path.join(MODEL_DIR, 'prepare_image.py')
RESULT_JSON = os.path.join(MODEL_DIR, 'result.json')
TMP_JPG = '/tmp/incoming.jpg'
LOG_FILE = '/home/xilinx/results.jsonl'

FIRESTORE_URL = "https://firestore.googleapis.com/v1/projects/smartfarm-27c25/databases/(default)/documents/fpga_results"

GPIO = MMIO(0x40020000, 0x10000)
GPIO.write(0x04, 0x0)
GPIO.write(0x00, 0x0)

DISEASE_TO_NUM = {'0_': 1, '1_': 2, '2_': 3, '3_': 4, '4_': 5}

def leds_off():
    GPIO.write(0x00, 0x0)

def disease_num(class_name):
    for prefix, n in DISEASE_TO_NUM.items():
        if class_name.startswith(prefix):
            return n
    return 0

processing_active = False
def processing_animation():
    i = 0
    while processing_active:
        GPIO.write(0x00, 1 << i)
        i = (i + 1) % 4
        time.sleep(0.2)
    leds_off()

def start_processing_anim():
    global processing_active
    processing_active = True
    threading.Thread(target=processing_animation, daemon=True).start()

def stop_processing_anim():
    global processing_active
    processing_active = False
    time.sleep(0.25)

inference_lock = threading.Lock()

print("[LED] startup sequence: 1->5")
for n in range(1, 6):
    GPIO.write(0x00, n & 0b1111)
    time.sleep(0.3)
leds_off()

def run_fpga_inference(jpeg_bytes):
    with open(TMP_JPG, 'wb') as f:
        f.write(jpeg_bytes)
    subprocess.run(['python3', PREPARE_PY, TMP_JPG], cwd=MODEL_DIR, check=True)
    if os.path.exists('/tmp/infer_done'):
        os.remove('/tmp/infer_done')
    open('/tmp/infer_request', 'w').close()
    timeout = 180
    start = time.time()
    while not os.path.exists('/tmp/infer_done'):
        if time.time() - start > timeout:
            raise TimeoutError("FPGA inference timeout")
        time.sleep(0.2)
    os.remove('/tmp/infer_done')
    with open(RESULT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

def jpeg_to_thumb_base64(jpeg_bytes, max_size=400, quality=80):
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.convert('RGB')
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        return base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception as e:
        print(f"  [WARN] thumbnail failed: {e}")
        return ""

latest_result = None

class ImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_result
        if self.path == '/':
            html = b'''<html><head><meta http-equiv="refresh" content="3"></head>
<body style="background:#111;text-align:center;color:white">
<h2>ESP32-CAM Live</h2>
<img src="/latest.jpg" style="max-width:480px">
<p style="color:#aaa">3s auto refresh</p>
</body></html>'''
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html)
        elif self.path == '/latest.jpg':
            if os.path.exists(TMP_JPG):
                with open(TMP_JPG, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path == '/result':
            if latest_result is not None:
                resp = latest_result
                latest_result = None
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(resp.encode('utf-8'))
            else:
                self.send_response(202)
                self.send_header('Connection', 'close')
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global latest_result
        if self.path not in ('/upload', '/disease'):
            self.send_response(404); self.end_headers()
            self.wfile.write(b'Not Found'); return
        print(f"\n[POST] path={self.path}")
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400); self.end_headers()
            self.wfile.write(b'Empty body'); return
        body = b''
        remaining = content_length
        while remaining > 0:
            chunk = self.rfile.read(min(4096, remaining))
            if not chunk:
                break
            body += chunk
            remaining -= len(chunk)
        if len(body) != content_length:
            self.send_response(400); self.end_headers()
            self.wfile.write(b'Incomplete'); return
        latest_result = None
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(b'OK')
        threading.Thread(target=self._run_inference, args=(body,), daemon=True).start()

    def _run_inference(self, body):
        global latest_result
        if not inference_lock.acquire(blocking=False):
            print("[SKIP] 이전 추론 진행 중 -> 이번 프레임 무시")
            return
        try:
            stop_processing_anim()
            leds_off()
            start_processing_anim()
            t0 = time.time()
            result = run_fpga_inference(body)
            elapsed = (time.time() - t0) * 1000
            top = result.get('top_class', '?')
            conf = result.get('confidence', 0)
            stop_processing_anim()
            d_n = disease_num(top)
            GPIO.write(0x00, d_n & 0b1111)
            print(f"[INFER] {len(body):,}B | {elapsed:.0f}ms")
            print(f"  -> {top} ({conf:.3f}) | LED={d_n}")
            latest_result = f"{top},{conf:.4f}"
            thumb_b64 = jpeg_to_thumb_base64(body)
            log_entry = {
                'timestamp': time.time(),
                'size_bytes': len(body),
                'elapsed_ms': elapsed,
                'top_class': top,
                'confidence': conf,
                'all_probs': result.get('all_probs', {}),
                'image_base64': thumb_b64,
            }
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            # Firestore REST API 전송
            try:
                fs_doc = {
                    "fields": {
                        "timestamp": {"timestampValue": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                        "top_class": {"stringValue": top},
                        "confidence": {"doubleValue": conf},
                        "elapsed_ms": {"doubleValue": elapsed},
                        "size_bytes": {"integerValue": str(len(body))},
                        "imageUrl": {"stringValue": "data:image/jpeg;base64," + thumb_b64},
                    }
                }
                all_probs = result.get('all_probs', {})
                map_fields = {}
                for k, v in all_probs.items():
                    map_fields[k] = {"doubleValue": v}
                fs_doc["fields"]["all_probs"] = {"mapValue": {"fields": map_fields}}
                data = json.dumps(fs_doc).encode('utf-8')
                req = urllib.request.Request(FIRESTORE_URL, data=data,
                                             headers={'Content-Type': 'application/json'},
                                             method='POST')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    print(f"  [FIREBASE] 전송 완료: {resp.status}")
            except Exception as e:
                print(f"  [FIREBASE] 전송 실패: {e}")
        except Exception as e:
            print(f"[ERROR] {e}")
            latest_result = "ERROR,0.0000"
            stop_processing_anim()
            leds_off()
        finally:
            inference_lock.release()

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print(f"[SERVER] FPGA inference server on port {SERVER_PORT}")
    print(f"[SERVER] Mode: ASYNC + POLL (POST /upload, GET /result)")
    print(f"[SERVER] Live view: GET http://<ip>:{SERVER_PORT}/")
    print(f"[SERVER] Firebase: fpga_results collection (REST API)")
    server = HTTPServer(('0.0.0.0', SERVER_PORT), ImageHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_processing_anim()
        leds_off()
        server.server_close()
