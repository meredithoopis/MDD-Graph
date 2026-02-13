import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor
import pandas as pd


dict_vocab = {"t": 0, "uw": 1, "er": 2, "ah": 3, "sh": 4, "ng": 5, "ow": 6, "aw": 7, "aa": 8, "th": 9, "ih": 10, "zh": 11, "k": 12, "y": 13, "l": 14, "uh": 15, "ch": 16, "w": 17, "b": 18, "v": 19, "ao": 20, "s": 21, "p": 22, "iy": 23, "r": 24, "eh": 25, "f": 26, "n": 27, "ay": 28, "oy": 29, "d": 30, "g": 31, "ey": 32, "err": 33, "dh": 34, "ae": 35, "hh": 36, "m": 37, "jh": 38, "z": 39, "<eps>": 40, "<blank>": 41}

PAD_ID = dict_vocab["<eps>"]
BLANK_ID = dict_vocab["<blank>"]
VOCAB_SIZE = len(dict_vocab)

L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]
L1_TO_ID = {l: i for i, l in enumerate(L1_LIST)}
NUM_L1 = len(L1_LIST)

def text_to_tensor(string_text):
    text = string_text.split(" ")
    return [dict_vocab[t] for t in text]

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    padding_side='right',
    do_normalize=True,
    return_attention_mask=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MDD_Dataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        waveform, _ = librosa.load("EN_MDD/WAV/" + row["Path"] + ".wav", sr=16000)

        canonical_ids  = text_to_tensor(row["Canonical"])
        transcript_ids = text_to_tensor(row["Transcript"])

        l1 = str(row["L1"]).lower()
        l1_id = L1_TO_ID.get(l1, 0)  

        return waveform, canonical_ids, transcript_ids, l1_id

    def __len__(self):
        return len(self.data)

def collate_fn(batch, pad_id=PAD_ID):
    wavs, canonicals, transcripts, l1_ids = zip(*batch)

    max_wav = max(len(w) for w in wavs)
    padded_wavs = [np.pad(w, (0, max_wav - len(w))) for w in wavs]

    inputs = feature_extractor(padded_wavs, sampling_rate=16000)
    input_values = torch.tensor(inputs.input_values).float().to(device)  # (B,T)

    max_can = max(len(c) for c in canonicals)
    canonical_pad = [c + [pad_id] * (max_can - len(c)) for c in canonicals]
    canonical = torch.tensor(canonical_pad).long().to(device)  # (B, T_can)

    transcript_flat = []
    transcript_lengths = []
    for t in transcripts:
        transcript_flat.extend(t)
        transcript_lengths.append(len(t))

    transcript_flat = torch.tensor(transcript_flat).long().to(device)
    transcript_lengths = torch.tensor(transcript_lengths).long().to(device)

    l1_ids = torch.tensor(l1_ids).long().to(device)  # (B,)

    return input_values, canonical, transcript_flat, transcript_lengths, l1_ids







