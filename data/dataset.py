import glob
import os
import pickle
import xml.etree.ElementTree as ET

import cv2
import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset


class MultimodalDataset:

    def __init__(self, subject_list, data_path, video_path, trial_indices=None, window_size=5, fs=256):

        self.subject_list = subject_list
        self.data_path = data_path
        self.video_path = video_path
        self.trial_indices = trial_indices

        self.window_size = window_size
        self.fs = fs
        self.window_samples = window_size * fs

        self.samples = []

        # ----------------------------------------
        # PREPROCESS INDEX (VERY IMPORTANT)
        # ----------------------------------------
        for subject in subject_list:

            with open(f"{data_path}/{subject}.dat", "rb") as f:
                d = pickle.load(f, encoding="latin1")

            eeg = d["data"][:, 0:32, :]
            ppg, eda, tmp = d["data"][:, 37, :], d["data"][:, 38, :], d["data"][:, 39, :]
            labels = d["labels"]

            num_trials = eeg.shape[0]

            if trial_indices is None:
                trials = range(num_trials)
            else:
                trials = trial_indices

            for t in trials:

                T = eeg.shape[-1]

                # sliding windows per trial
                for start in range(0, T - self.window_samples, self.window_samples):

                    self.samples.append({
                        "subject": subject,
                        "trial": t,
                        "start": start
                    })

            # cache reference
            setattr(self, f"{subject}_data", {
                "eeg": eeg,
                "ppg": ppg,
                "eda": eda,
                "tmp": tmp,
                "labels": labels
            })

    def load_video_window(self, subject, trial, start, length):
        start = int((start/128)*50)
        length = int((length/128)*50)

        path = f"{self.video_path}/{subject}/{subject}_trial{trial+1:02d}.avi"

        cap = cv2.VideoCapture(path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start >= total_frames:
            raise ValueError(
                f"Start beyond video length\n"
                f"path={path}\nstart={start}\nframes={total_frames}"
            )

        if start + length > total_frames:
            length = total_frames - start

        # IMPORTANT: safe seek
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))

        for i in range(length):

            if i % 5 != 0:
                continue

            ret, frame = cap.read()
            frame = cv2.resize(frame, (112, 112))

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            yield frame

        cap.release()
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]

        subject = s["subject"]
        trial = s["trial"]
        start = s["start"]

        data = getattr(self, f"{subject}_data")

        eeg = torch.from_numpy(data["eeg"][trial, :, start:start+self.window_samples]).float()
        ppg = torch.from_numpy(data["ppg"][trial, start:start+self.window_samples]).float()
        eda = torch.from_numpy(data["eda"][trial, start:start+self.window_samples]).float()
        tmp = torch.from_numpy(data["tmp"][trial, start:start+self.window_samples]).float()


        video_path = f"{self.video_path}/{subject}/{subject}_trial{trial+1:02d}.avi"

        if os.path.exists(video_path):

            video = self.load_video_window(subject, trial, start, self.window_samples)

            label = {
                "valence": torch.tensor(data["labels"][trial][0] >= 5.0, dtype=torch.float),
                "arousal": torch.tensor(data["labels"][trial][1] >= 5.0, dtype=torch.float)
            }


            q_eeg = torch.diff(eeg, dim=-1).std(dim=-1).mean(dim=-1)
            q_ppg = torch.diff(ppg, dim=-1).std(dim=-1)
            q_eda = torch.diff(eda, dim=-1).abs().mean(dim=-1)
            q_tmp = torch.diff(tmp, dim=-1).abs().mean(dim=-1)
            q_video = 0.0 # torch.diff(video_feat, dim=1).norm(dim=-1).mean(dim=-1)

            quality_vector = torch.tensor([q_video, q_eeg, q_ppg, q_eda, q_tmp], dtype=torch.float)


            return {
                "face": video,  # Python Generator object
                "eeg": eeg,  # [32, T]
                "ppg": ppg,  # [T]
                "eda": eda,  # [T]
                "tmp": tmp,  # [T]
                "targets": label,
                "signal_quality": quality_vector,  # Tensor [M,]
                "subject": subject
            }

        return None


def resample_fixed(data_chunk, target_length):
    """
    Resample along time axis.
    Input:
        [T]
        [T,C]
    Output:
        [target_length]
        [target_length,C]
    """

    if len(data_chunk) == 0:
        return np.zeros((target_length,), dtype=np.float32)

    return resample(data_chunk, target_length, axis=0)

def normalize(tensor):
    # tensor shape: [Channels, Time]
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True) + 1e-8
    return (tensor - mean) / std


class MAHNOBMultimodalDataset(Dataset):

    def __init__(
            self,
            base_path,
            subjects_to_keep,
            window_sec=30,
            stride_sec=15,
            len_eeg=256 * 30,
            len_ecg=256 * 30,
            len_eda=256 * 30,
            len_tmp=256 * 30,
            len_rsp=256 * 30,
            len_eye=60 * 30
    ):

        self.base_path = base_path
        self.subjects_to_keep = subjects_to_keep

        self.window_sec = window_sec
        self.stride_sec = stride_sec

        self.lengths = {
            "eeg": len_eeg,
            "ecg": len_ecg,
            "eda": len_eda,
            "tmp": len_tmp,
            "rsp": len_rsp,
            "eye": len_eye
        }

        self.all_samples = []

        search_pattern = os.path.join(base_path, "*", "session.xml")
        xml_files = glob.glob(search_pattern)

        if len(xml_files) == 0:
            raise ValueError(
                f"No session.xml files found under {base_path}"
            )

        print(f"Found {len(xml_files)} sessions.")

        for xml_path in xml_files:

            folder_path = os.path.dirname(xml_path)

            try:
                root = ET.parse(xml_path).getroot()

                valence, arousal = int(root.attrib["feltVlnc"]), int(root.attrib["feltArsl"])

                if valence == 0 or arousal == 0:
                    continue

                subject_id = int(root.find("subject").attrib["id"])
                session_id = int(root.attrib["sessionId"])
                labels = {
                    "valence": float(valence >= 5),
                    "arousal": float(arousal >= 5)
                }

            except Exception:
                continue

            if subject_id not in subjects_to_keep:
                continue

            data_dict, labels = self._process_session(folder_path, labels)

            if data_dict is None:
                continue

            n_windows = len(labels)

            for i in range(n_windows):

                self.all_samples.append({

                    "eeg": data_dict["eeg"][i],
                    "ecg": data_dict["ecg"][i],
                    "eda": data_dict["eda"][i],
                    "tmp": data_dict["tmp"][i],
                    "rsp": data_dict["rsp"][i],
                    "eye": data_dict["eye"][i],

                    "signal_quality":
                        data_dict["signal_quality"][i],

                    "label": labels[i],

                    "subject_id": subject_id,
                    "session_id": session_id
                })

        print(
            f"Loaded {len(self.all_samples)} windows "
            f"from {len(subjects_to_keep)} subjects."
        )

    def _process_session(self, folder_path, label):

        bdf_files = glob.glob(os.path.join(folder_path, "*.bdf"))

        eye_files = glob.glob(
            os.path.join(folder_path,
                         "*All-Data*Section_*.tsv")
        )

        if len(bdf_files) == 0:
            print(folder_path, "NO BDF")

        if len(eye_files) == 0:
            print(folder_path, "NO EYE")

        if (
                len(bdf_files) == 0 or
                len(eye_files) == 0
        ):
            return None, None

        #################################
        # EYE
        #################################

        try:

            header_row = 0

            with open(
                    eye_files[0],
                    "r",
                    encoding="utf-8",
                    errors="ignore"
            ) as f:

                for i, line in enumerate(f):

                    if (
                            "Timestamp" in line and
                            "GazePoint" in line
                    ):
                        header_row = i
                        break

            df_eye = pd.read_csv(
                eye_files[0],
                sep="\t",
                skiprows=header_row
            )

            pupil = df_eye[
                [c for c in df_eye.columns
                 if "Pupil" in c][0]
            ].values

            gx = df_eye[
                [c for c in df_eye.columns
                 if "GazePointX" in c][0]
            ].values

            gy = df_eye[
                [c for c in df_eye.columns
                 if "GazePointY" in c][0]
            ].values

            raw_eye = np.nan_to_num(
                np.column_stack((pupil, gx, gy))
            )

        except Exception:
            return None, None

        #################################
        # PHYSIO
        #################################

        try:

            raw = mne.io.read_raw_bdf(
                bdf_files[0],
                preload=True,
                verbose=False
            )

            fs_physio = raw.info["sfreq"]
            fs_eye = 60.0

            eeg = raw.copy().pick(
                raw.ch_names[:32]
            ).get_data().T

            ecg = raw.copy().pick(
                ["EXG1", "EXG2", "EXG3"]
            ).get_data().T

            eda = raw.copy().pick(
                ["GSR1"]
            ).get_data().T

            rsp = raw.copy().pick(
                ["Resp"]
            ).get_data().T

            tmp = raw.copy().pick(
                ["Temp"]
            ).get_data().T

        except Exception:
            return None, None

        #################################
        # WINDOWING
        #################################

        duration = min(
            len(eeg) / fs_physio,
            len(raw_eye) / fs_eye
        )

        data_dict = {
            "eeg": [],
            "ecg": [],
            "eda": [],
            "tmp": [],
            "rsp": [],
            "eye": [],
            "signal_quality": []
        }

        labels = []

        for t in np.arange(
                0,
                duration - self.window_sec,
                self.stride_sec
        ):

            p_start = int(t * fs_physio)
            p_end = int((t + self.window_sec) * fs_physio)

            e_start = int(t * fs_eye)
            e_end = int((t + self.window_sec) * fs_eye)

            eeg_chunk = eeg[p_start:p_end]
            eye_chunk = raw_eye[e_start:e_end]

            if len(eeg_chunk) == 0:
                continue

            if len(eye_chunk) == 0:
                continue

            ecg_chunk = ecg[p_start:p_end]
            eda_chunk = eda[p_start:p_end]
            tmp_chunk = tmp[p_start:p_end]
            rsp_chunk = rsp[p_start:p_end]

            eeg_t = torch.from_numpy(eeg_chunk).float()
            ecg_t = torch.from_numpy(ecg_chunk).float()
            eda_t = torch.from_numpy(eda_chunk).float()
            rsp_t = torch.from_numpy(rsp_chunk).float()
            tmp_t = torch.from_numpy(tmp_chunk).float()
            eye_t = torch.from_numpy(eye_chunk).float()

            #################################
            # SIGNAL QUALITY
            #################################

            q_eeg = torch.diff(eeg_t, dim=0).std(dim=0).mean()

            q_ecg = torch.diff(ecg_t, dim=0).std(dim=0).mean()

            q_eda = torch.diff(eda_t, dim=0).abs().mean()

            q_rsp = torch.diff(rsp_t, dim=0).std(dim=0).mean()

            q_tmp = torch.diff(tmp_t, dim=0).abs().mean()

            q_eye = torch.diff(eye_t, dim=0).std(dim=0).mean()

            quality = np.array([
                q_eeg,
                q_ecg,
                q_eda,
                q_rsp,
                q_tmp,
                q_eye
            ], dtype=np.float32)

            #################################
            # RESAMPLING
            #################################

            data_dict["eeg"].append(
                resample_fixed(
                    eeg_t,
                    self.lengths["eeg"]
                )
            )

            data_dict["ecg"].append(
                resample_fixed(
                    ecg_chunk,
                    self.lengths["ecg"]
                )
            )

            data_dict["eda"].append(
                resample_fixed(
                    eda_chunk,
                    self.lengths["eda"]
                )
            )

            data_dict["tmp"].append(
                resample_fixed(
                    tmp_chunk,
                    self.lengths["tmp"]
                )
            )

            data_dict["rsp"].append(
                resample_fixed(
                    rsp_chunk,
                    self.lengths["rsp"]
                )
            )

            data_dict["eye"].append(
                resample_fixed(
                    eye_chunk,
                    self.lengths["eye"]
                )
            )

            data_dict["signal_quality"].append(
                quality
            )

            labels.append(label)

        return data_dict, labels

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):

        sample = self.all_samples[idx]

        return {

        "eeg":
            normalize(torch.from_numpy(
                sample["eeg"]
            ).float().T),

        "ecg":
            normalize(torch.from_numpy(
                sample["ecg"]
            ).float().T),

        "eda":
            normalize(torch.from_numpy(
                sample["eda"]
            ).float().T),

        "tmp":
            normalize(torch.from_numpy(
                sample["tmp"]
            ).float().T),

        "rsp":
            normalize(torch.from_numpy(
                sample["rsp"]
            ).float().T),

        "eye":
            normalize(torch.from_numpy(
                sample["eye"]
            ).float().T),

        "signal_quality":
            torch.from_numpy(
                sample["signal_quality"]
            ).float(),

        "targets": {
            "valence": torch.tensor(sample["label"]["valence"], dtype=torch.float),
            "arousal": torch.tensor(sample["label"]["arousal"], dtype=torch.float)
        },

        "subject":
            sample["subject_id"],

        "session":
            sample["session_id"]
    }