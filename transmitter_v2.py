import asyncio
import logging
import os
import queue
import threading
import numpy as np
import sounddevice as sd
import webrtcvad

os.environ["WHISPER_PRINT_PROGRESS"] = "0"
from pywhispercpp.model import Model
from bless import BlessServer, GATTCharacteristicProperties, GATTAttributePermissions

MIC_SR = 48000
TARGET_SR = 16000
DOWNSAMPLE = MIC_SR // TARGET_SR
FRAME_MS = 30
FRAME_SAMPLES_MIC = MIC_SR * FRAME_MS // 1000
SILENCE_END_FRAMES = 25
MIN_SPEECH_FRAMES = 15
MAX_SPEECH_FRAMES = 300
MIC_DEVICE = None

SERVICE_UUID = "0000A100-0000-1000-8000-00805F9B34FB"
CHAR_UUID    = "0000A101-0000-1000-8000-00805F9B34FB"
DEVICE_NAME  = "EdgeWhisper-TX"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ew")

audio_queue = queue.Queue()
text_queue = None
LATEST_TEXT = b"EdgeWhisper ready"


def audio_callback(indata, frames, time, status):
    arr = np.frombuffer(bytes(indata), dtype=np.int16)
    downsampled = arr[::DOWNSAMPLE].tobytes()
    audio_queue.put(downsampled)


def stt_worker(loop):
    log.info("Loading Whisper model...")
    model = Model("model_tiny.bin", n_threads=4, print_progress=False,
                  print_realtime=False, print_timestamps=False)
    vad = webrtcvad.Vad(3)
    log.info("STT ready.")

    speech_buffer = bytearray()
    silence_frames = 0
    speech_frames = 0
    in_speech = False

    while True:
        frame = audio_queue.get()
        is_speech = vad.is_speech(frame, TARGET_SR)

        if is_speech:
            speech_buffer.extend(frame)
            speech_frames += 1
            silence_frames = 0
            in_speech = True
            if speech_frames >= MAX_SPEECH_FRAMES:
                silence_frames = SILENCE_END_FRAMES
        elif in_speech:
            speech_buffer.extend(frame)
            silence_frames += 1
            if silence_frames >= SILENCE_END_FRAMES:
                if speech_frames >= MIN_SPEECH_FRAMES:
                    with audio_queue.mutex:
                        audio_queue.queue.clear()
                    audio_np = (
                        np.frombuffer(bytes(speech_buffer), dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                    segments = model.transcribe(audio_np)
                    text = " ".join(s.text.strip() for s in segments).strip()
                    if text and not text.startswith("(") and "[BLANK_AUDIO]" not in text:
                        log.info("TX: " + text)
                        asyncio.run_coroutine_threadsafe(text_queue.put(text), loop)
                speech_buffer.clear()
                silence_frames = 0
                speech_frames = 0
                in_speech = False


def read_request(characteristic, **kwargs):
    return LATEST_TEXT


def write_request(characteristic, value, **kwargs):
    characteristic.value = value


async def run_ble(loop):
    global LATEST_TEXT
    server = BlessServer(name=DEVICE_NAME, loop=loop)
    server.read_request_func = read_request
    server.write_request_func = write_request
    await server.add_new_service(SERVICE_UUID)
    flags = GATTCharacteristicProperties.read | GATTCharacteristicProperties.notify
    perms = GATTAttributePermissions.readable
    await server.add_new_characteristic(SERVICE_UUID, CHAR_UUID, flags, b"", perms)
    await server.start()
    log.info("BLE '" + DEVICE_NAME + "' advertising.")

    while True:
        text = await text_queue.get()
        payload = text.encode("utf-8")
        LATEST_TEXT = payload
        for i in range(0, len(payload), 180):
            piece = payload[i:i+180]
            server.get_characteristic(CHAR_UUID).value = piece
            server.update_value(SERVICE_UUID, CHAR_UUID)
            await asyncio.sleep(0.02)


async def main():
    global text_queue
    text_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    threading.Thread(target=stt_worker, args=(loop,), daemon=True).start()
    with sd.RawInputStream(samplerate=MIC_SR, blocksize=FRAME_SAMPLES_MIC,
                           device=MIC_DEVICE, dtype="int16", channels=1,
                           callback=audio_callback):
        await run_ble(loop)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
