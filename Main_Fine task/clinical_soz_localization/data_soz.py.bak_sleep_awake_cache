import os

from collections import OrderedDict

import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset



try:

    import mne

except Exception:

    mne = None



try:

    from scipy.signal import resample_poly

except Exception:

    resample_poly = None





def read_raw_edf_safe(path):

    if mne is None:

        raise ImportError("mne is required for EDF reading.")

    try:

        return mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")

    except UnicodeDecodeError:

        return mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR", encoding="latin1")





def match_channel(raw_names, ch_name):

    if ch_name in raw_names:

        return ch_name



    target = str(ch_name).strip().lower()

    mapping = {str(c).strip().lower(): c for c in raw_names}



    if target in mapping:

        return mapping[target]



    target2 = target.replace(" ", "").replace("-", "").replace("_", "")

    for c in raw_names:

        c2 = str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

        if c2 == target2:

            return c



    raise KeyError("Channel not found: {}. Example channels: {}".format(ch_name, raw_names[:10]))





def resample_1d(x, fs_in, fs_out):

    fs_in = float(fs_in)

    fs_out = float(fs_out)



    if abs(fs_in - fs_out) < 1e-6:

        return x.astype(np.float32)



    if resample_poly is not None:

        from math import gcd

        a = int(round(fs_out))

        b = int(round(fs_in))

        g = gcd(a, b)

        return resample_poly(x, a // g, b // g).astype(np.float32)



    old_t = np.arange(len(x)) / fs_in

    new_n = int(round(len(x) * fs_out / fs_in))

    new_t = np.arange(new_n) / fs_out

    return np.interp(new_t, old_t, x).astype(np.float32)





def robust_zscore(x):

    x = np.asarray(x, dtype=np.float32)

    med = np.nanmedian(x)

    mad = np.nanmedian(np.abs(x - med))

    scale = 1.4826 * mad



    if not np.isfinite(scale) or scale < 1e-6:

        scale = np.nanstd(x)



    if not np.isfinite(scale) or scale < 1e-6:

        scale = 1.0



    x = (x - med) / scale

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.clip(x, -10, 10)

    return x.astype(np.float32)





class OmniSOZDataset(Dataset):

    def __init__(

        self,

        manifest_csv,

        split=None,

        split_col="split_internal",

        context_sec=60,

        patch_sec=1,

        target_fs=256,

        max_channels=128,

        max_cache=2,

    ):

        self.manifest_csv = str(manifest_csv)

        self.df = pd.read_csv(manifest_csv)



        self.split_col = split_col



        if split is not None:

            if self.split_col not in self.df.columns:

                if "split" in self.df.columns:

                    self.split_col = "split"

                else:

                    raise ValueError("No split column found in manifest.")



            self.df = self.df[

                self.df[self.split_col].astype(str).str.lower() == str(split).lower()

            ].reset_index(drop=True)



        self.context_sec = float(context_sec)

        self.patch_sec = float(patch_sec)

        self.target_fs = float(target_fs)

        self.max_channels = int(max_channels)

        self.seq_len = int(round(self.context_sec / self.patch_sec))

        self.patch_len = int(round(self.patch_sec * self.target_fs))

        self.target_len = int(self.seq_len * self.patch_len)

        self.max_cache = int(max_cache)

        self._cache = OrderedDict()



        if len(self.df) == 0:

            raise RuntimeError("Empty dataset for split={} from {}".format(split, manifest_csv))



    def __len__(self):

        return len(self.df)



    def _get_raw(self, edf_path):

        edf_path = str(edf_path)



        if edf_path in self._cache:

            raw = self._cache.pop(edf_path)

            self._cache[edf_path] = raw

            return raw



        raw = read_raw_edf_safe(edf_path)

        self._cache[edf_path] = raw



        while len(self._cache) > self.max_cache:

            _, old = self._cache.popitem(last=False)

            try:

                old.close()

            except Exception:

                pass



        return raw



    def __getitem__(self, idx):

        r = self.df.iloc[idx]



        edf_path = r["edf_path"]

        ch_name = r["channel_name"]

        y = int(r["label"])



        start_sec = float(r["segment_start_sec"])

        end_sec = float(r["segment_end_sec"])



        cache_path = None
        if "cache_path" in r.index:
            cp = r["cache_path"]
            if isinstance(cp, str) and len(cp) > 0 and os.path.exists(cp):
                cache_path = cp

        if cache_path is not None:
            x = np.load(cache_path).astype(np.float32)
        else:
            raw = self._get_raw(edf_path)

            raw_ch = match_channel(raw.ch_names, ch_name)

            fs_in = float(raw.info["sfreq"])

            start = int(round(start_sec * fs_in))

            stop = int(round(end_sec * fs_in))

            x = raw.get_data(picks=[raw_ch], start=start, stop=stop)[0]

            x = resample_1d(x, fs_in=fs_in, fs_out=self.target_fs)



        if len(x) < self.target_len:

            pad = np.zeros(self.target_len, dtype=np.float32)

            pad[:len(x)] = x

            x = pad

        elif len(x) > self.target_len:

            x = x[:self.target_len]



        x = robust_zscore(x)



        # Convert one long 1D segment into patch format expected by SABrant:

        # [channels, seq_len, patch_len]

        x_patch = x.reshape(self.seq_len, self.patch_len)



        x_pad = np.zeros((self.max_channels, self.seq_len, self.patch_len), dtype=np.float32)

        mask = np.zeros((self.max_channels,), dtype=np.float32)



        x_pad[0, :, :] = x_patch

        mask[0] = 1.0



        return {

            "x": torch.from_numpy(x_pad),

            "label": torch.tensor(y, dtype=torch.long),

            "fs": torch.tensor(self.target_fs, dtype=torch.float32),

            "channel_mask": torch.from_numpy(mask),

            "patient_name": str(r.get("patient_name", "")),

            "edf_name": str(r.get("edf_name", "")),

            "channel_name": str(r.get("channel_name", "")),

            "anatomical": str(r.get("anatomical", "")),

            "soz": int(r.get("soz", 0)),

            "resection": int(r.get("resection", 0)),

            "outcome": str(r.get("outcome", "")),

            "segment_start_sec": float(r.get("segment_start_sec", 0.0)),

            "segment_end_sec": float(r.get("segment_end_sec", 0.0)),

        }

