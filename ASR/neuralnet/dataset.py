import torch
import torchaudio
import torch.nn as nn
import pandas as pd
from utils import TextProcess


# NOTE: add time stretch
class SpecAugment(nn.Module):

    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sample_rate, n_mels=n_mels,
                            win_length=win_length, hop_length=hop_length)

    def forward(self, x):
        x = self.transform(x)  # mel spectrogram
        x = torch.log(x + 1e-14)  # logrithmic, add small value to avoid inf
        return x


def get_featurizer(sample_rate, n_feats=81):
    return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80)


class Data(torch.utils.data.Dataset):

    # this makes it easier to be ovveride in argparse
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15,
        "max_spec_len": 1650, # maximum allowed time‑bins
        "max_channels":  1,      # only mono audio
    }

    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy,
                time_mask, freq_mask, max_spec_len, max_channels, valid=False, shuffle=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()
        self.max_spec_len  = max_spec_len
        self.max_channels  = max_channels
                    
        print("Loading data json file from", json_path)
        self.data = pd.read_json(json_path, lines=True)

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        original_idx = idx
        max_attempts = len(self.data)
        attempts = 0

        while attempts < max_attempts:
            try:
                row = self.data.iloc[idx]
                file_path = row["key"]
                waveform, _ = torchaudio.load(file_path)
                label_str = row["text"]
                label = self.text_process.text_to_int_sequence(label_str)
                spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
                spec_len = spectrogram.shape[-1] // 2
                label_len = len(label)
                # after computing spec_len, label_len, etc.
                max_len    = self.max_spec_len
                max_ch     = self.max_channels

                if spec_len < label_len:
                    raise Exception("spectrogram length < label length")
                if spectrogram.shape[0] > max_ch:
                    raise Exception(f"too many channels ({spectrogram.shape[0]}); expected ≤{max_ch}")
                if spectrogram.shape[2] > max_len:
                    raise Exception(f"spectrogram too long ({spectrogram.shape[2]}); max is {max_len}")
                if label_len == 0:
                    raise Exception('label len is zero... skipping %s'%file_path)
                return spectrogram, label, spec_len, label_len
            
            except Exception as e:
                if self.log_ex:
                    fp = locals().get("file_path", "Unknown")
                    print(f"[idx={idx}] {e} – file: {fp}")
                # move to previous index if possible, else next    
                idx = idx - 1 if idx > 0 else idx + 1
                attempts += 1
        # if we exhaust the loop, no valid sample found
        raise RuntimeError(f"No valid sample found after {max_attempts} tries, starting at index {original_idx}")

    def describe(self):
        return self.data.describe()


def collate_fn_padd(data):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # print(data)
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
       # print(spectrogram.shape)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.tensor(label, dtype=torch.long))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = input_lengths
    # print(spectrograms.shape)
    label_lengths = label_lengths
    # ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return spectrograms, labels, input_lengths, label_lengths
