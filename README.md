# Edge Whisper

A fully offline, privacy-preserving voice assistant running on a Raspberry Pi 5.
No internet connection is required at any point. The system captures speech,
transcribes it locally using Whisper, generates a response using an on-device LLM,
and transmits the answer wirelessly to a phone via Bluetooth Low Energy (BLE).

**Team:** Santhoshkrishna Chezhian & Vatsalam Krishna Jha

---

## Pipeline

```
Microphone → Whisper STT → LLM (Llama 3.2 3B) → BLE GATT Server → Phone Display
```

---

## Motivation

Modern voice assistants (Siri, Alexa, Google Assistant) send audio to remote
servers for processing. This introduces latency, privacy risks, and a dependency
on internet connectivity. Edge Whisper demonstrates that the full pipeline —
speech recognition, language understanding, and wireless transmission — can run
entirely on an $80 embedded device.

---

## Hardware

- Raspberry Pi 5 (8GB RAM)
- USB microphone
- Smartphone with **nRF Connect** (iOS/Android) — free, by Nordic Semiconductor

---

## Repository Structure

```
edge-whisper/
├── transmitter_v1.py     # mic → Whisper STT → terminal output
├── transmitter_v2.py     # mic → Whisper STT → BLE → phone
├── transmitter_v3.py     # mic → Whisper STT → LLM → BLE → phone  (main demo)
├── benchmark_llms.py     # evaluates 3 LLMs on 15 fixed questions
├── score_answers.py      # interactive quality scoring tool for benchmark results
├── test_llm.py           # standalone LLM sanity check
├── requirements.txt
└── results/
    └── benchmark.csv     # benchmark output (3 models × 15 questions)
```

---

## Setup

### 1. System packages

```bash
sudo apt update && sudo apt install -y \
    git build-essential cmake \
    python3-pip python3-venv python3-dev \
    portaudio19-dev libopenblas-dev \
    bluetooth bluez libbluetooth-dev \
    alsa-utils ffmpeg sox
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Whisper model

Build whisper.cpp and download the `tiny.en` model:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build && cmake --build build --config Release -j
bash ./models/download-ggml-model.sh tiny.en
cp models/ggml-tiny.en.bin ../model_tiny.bin
cd ..
```

### 4. LLM models

Download into the `models/` directory (not tracked by git — files are too large):

```bash
mkdir -p models

# Llama 3.2 1B — fastest (~770MB)
wget -O models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Llama 3.2 3B — recommended (~1.9GB)
wget -O models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Phi-3 Mini — Microsoft architecture (~2.3GB)
wget -O models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf
```

---

## Usage

### Basic speech-to-text (terminal output)

```bash
source venv/bin/activate
python transmitter_v1.py
```

Speak into the microphone. Transcribed text prints to the terminal after each pause.

### Speech-to-text with Bluetooth

```bash
source venv/bin/activate
sudo bluetoothctl discoverable-timeout 0
sudo bluetoothctl discoverable on
python transmitter_v2.py
```

On your phone, open **nRF Connect**, scan for `EdgeWhisper-TX`, connect, expand
service `0000A100-...`, tap characteristic `0000A101-...`, enable notifications,
and set display format to UTF-8.

### Full voice assistant (main demo)

```bash
source venv/bin/activate
python transmitter_v3.py
```

Speak a question. The Pi transcribes it with Whisper, generates an answer with
Llama 3.2 3B, and sends both question and answer to your phone via BLE.

---

## BLE Configuration

| Parameter | Value |
|---|---|
| Device name | `EdgeWhisper-TX` |
| Service UUID | `0000A100-0000-1000-8000-00805F9B34FB` |
| Characteristic UUID | `0000A101-0000-1000-8000-00805F9B34FB` |

---

## Benchmark

The benchmark compares three LLMs on 15 fixed questions across 5 categories
(factual recall, math, definitions, common sense, conversational).

```bash
python benchmark_llms.py        # runs all 3 models, saves results/benchmark.csv
python score_answers.py         # interactive quality scoring (1-5 per answer)
```

### Preliminary results (Raspberry Pi 5, 8GB RAM)

| Model | Avg response time | Avg tokens/sec | Peak RAM |
|---|---|---|---|
| Llama 3.2 1B Q4 | 2.16s | 12.0 | 1729 MB |
| Llama 3.2 3B Q4 | 5.18s | 5.0 | 4204 MB |
| Phi-3 Mini Q4 | 5.79s | 3.9 | 5384 MB |

Key observations:
- Llama 1B is 2.4× faster than 3B but makes arithmetic and ordering errors
- Phi-3 Mini uses 5.4GB RAM, approaching the Pi 5's 8GB limit
- All models handle language-based tasks (definitions, common sense) well

---

## Technical Notes

**Audio pipeline:** The USB mic captures at 48kHz (its native rate). The script
downsamples to 16kHz by taking every 3rd sample before passing audio to Whisper
and the VAD (Voice Activity Detector).

**Quantization:** LLMs are stored in Q4_K_M format — 3.21 billion parameters
compressed from 16-bit floats to ~4.5 bits per weight on average. This reduces
the Llama 3B model from ~6.4GB to 1.9GB with ~2% quality loss.

**Why Pi 5:** The Pi 5's memory bandwidth (~17 GB/s vs ~5 GB/s on Pi 4) is the
key enabler. LLM inference is memory-bound, not compute-bound. The Pi 5 achieves
~10 tokens/sec with Llama 3B, close to its theoretical bandwidth ceiling.

---

## License

MIT
