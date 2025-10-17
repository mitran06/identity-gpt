import os
import time
from pathlib import Path
from collections import deque, Counter
import queue
import numpy as np
import sounddevice as sd
import torch
from scipy import signal
from speechbrain.inference import EncoderClassifier

model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SEC = 0.8
HOP_SEC = 0.4
THRESHOLD = 0.50
SMOOTH_WINDOW = 5
EMBED_DIR = Path("embeddings")

print("Loading SpeechBrain encoder...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)
print("Encoder ready.")

def load_known_embeddings():
    known = {}
    if not EMBED_DIR.exists():
        return known
    for f in EMBED_DIR.iterdir():
        if f.suffix == ".npy":
            name = f.stem
            known[name] = np.load(f)
    return known

known_embeddings = load_known_embeddings()
if not known_embeddings:
    print("⚠️ No enrolled speakers found. Run enroll.py first.")
    exit(1)

q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print('Status:', status)
    q.put(indata.copy())

def resample_if_necessary(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down)

def tensor_from_chunk(chunk: np.ndarray) -> torch.Tensor:
    arr = chunk.squeeze()
    arr = resample_if_necessary(arr, SAMPLE_RATE, SAMPLE_RATE)
    tensor = torch.tensor(arr.astype(np.float32)).unsqueeze(0)
    return tensor

def compute_embedding_from_chunk(chunk: np.ndarray) -> np.ndarray:
    tensor = tensor_from_chunk(chunk)
    with torch.no_grad():
        emb = classifier.encode_batch(tensor)
    emb = emb.squeeze().cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb

def identify_speaker(emb: np.ndarray, knowns: dict):
    best_name = None
    best_score = -1.0
    for name, ref in knowns.items():
        score = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

def main_loop():
    chunk_size = int(CHUNK_SEC * SAMPLE_RATE)
    hop_size = int(HOP_SEC * SAMPLE_RATE)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=audio_callback)
    stream.start()

    buffer = np.zeros((0, CHANNELS), dtype=np.float32)
    recent_preds = deque(maxlen=SMOOTH_WINDOW)
    last_announced = None

    print(" Listening... (Ctrl+C to stop)")
    try:
        while True:
            try:
                data = q.get(timeout=1.0)
            except queue.Empty:
                continue

            buffer = np.concatenate((buffer, data), axis=0)

            while buffer.shape[0] >= chunk_size:
                window = buffer[:chunk_size]
                wav_tensor = torch.tensor(window.T).unsqueeze(0)
                speech_ts = get_speech_timestamps(wav_tensor, model, sampling_rate=SAMPLE_RATE)

                if speech_ts:
                    emb = compute_embedding_from_chunk(window)
                    name, score = identify_speaker(emb, known_embeddings)


                    if score >= THRESHOLD:
                        recent_preds.append((name, score))
                    else:
                        recent_preds.append(("Unknown", score))

                    names_only = [n for n, s in recent_preds]
                    if names_only:
                        most_common = Counter(names_only).most_common(1)[0][0]
                        if most_common != last_announced:
                            scores_for = [s for n, s in recent_preds if n == most_common]
                            avg_score = float(np.mean(scores_for)) if scores_for else 0.0
                            if most_common != "Unknown":
                                print(f"\n {most_common} is speaking (avg score={avg_score:.2f})")
                            else:
                                print("\n Unknown speaker detected")
                            last_announced = most_common

                buffer = buffer[hop_size:]

    except KeyboardInterrupt:
        print("\n Stopping...")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main_loop()
