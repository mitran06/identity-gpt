import os
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import torchaudio
import torch
import soundfile as sf
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 3
SAMPLES_PER_PERSON = 4
EMBED_DIR = "embeddings"
META_FILE = os.path.join(EMBED_DIR, "metadata.json")

os.makedirs(EMBED_DIR, exist_ok=True)

print("Loading encoder model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
print("Model loaded.")


def record_sample(filename, seconds=RECORD_SECONDS, fs=SAMPLE_RATE):
    print(f"Recording {seconds}s -> {filename} (speak now)...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=CHANNELS, dtype='int16')
    sd.wait()
    wavfile.write(filename, fs, audio)
    print("Saved:", filename)


def load_audio_for_speechbrain(path):
    path = os.path.normpath(path)
    try:
        audio, fs = sf.read(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {path}: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    return signal, fs


def compute_embedding(wav_path):
    signal, fs = load_audio_for_speechbrain(wav_path)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)
        emb = emb.squeeze().cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb


def enroll_person(name):
    person_dir = os.path.join(EMBED_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    sample_paths = []
    for i in range(SAMPLES_PER_PERSON):
        path = os.path.join(person_dir, f"{name}_{i+1}.wav")
        record_sample(path)
        sample_paths.append(path)

    embeddings = []
    for p in sample_paths:
        emb = compute_embedding(p)
        embeddings.append(emb)
    embeddings = np.stack(embeddings, axis=0)
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    emb_file = os.path.join(EMBED_DIR, f"{name}.npy")
    np.save(emb_file, mean_emb)
    print(f"Saved averaged embedding to {emb_file}")

    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta[name] = {"embedding_file": os.path.basename(emb_file)}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Enrolled {name} successfully.")


if __name__ == "__main__":
    print("Enroll")
    name = input("Enter the person's name to enroll: ").strip()
    if not name:
        print("Name required.")
    else:
        enroll_person(name)
