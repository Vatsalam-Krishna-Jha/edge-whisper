import queue
import numpy as np
import sounddevice as sd
import webrtcvad
from pywhispercpp.model import Model

MIC_SR = 48000
TARGET_SR = 16000
DOWNSAMPLE = MIC_SR // TARGET_SR
FRAME_MS = 30
FRAME_SAMPLES_MIC = MIC_SR * FRAME_MS // 1000
SILENCE_END_FRAMES = 25
MIN_SPEECH_FRAMES = 15
MAX_SPEECH_FRAMES = 300
MIC_DEVICE = None

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    arr = np.frombuffer(bytes(indata), dtype=np.int16)
    downsampled = arr[::DOWNSAMPLE].tobytes()
    audio_queue.put(downsampled)


def main():
    print("Loading Whisper model...")
    model = Model("model_tiny.bin", n_threads=4, print_progress=False,
                  print_realtime=False, print_timestamps=False)
    vad = webrtcvad.Vad(3)
    print("Ready. Speak into the mic. Pause ~1s after each sentence. Ctrl+C to quit.\n")

    speech_buffer = bytearray()
    silence_frames = 0
    speech_frames = 0
    in_speech = False

    with sd.RawInputStream(samplerate=MIC_SR, blocksize=FRAME_SAMPLES_MIC,
                           device=MIC_DEVICE, dtype="int16", channels=1,
                           callback=audio_callback):
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
                        if text and not text.startswith("("):
                            print(f">>> {text}")
                    speech_buffer.clear()
                    silence_frames = 0
                    speech_frames = 0
                    in_speech = False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
