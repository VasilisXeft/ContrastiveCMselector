import os
import pickle

import cv2
import torch


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

            q_eeg = 1 / (eeg.std(dim=-1).mean(dim=-1) + 1e-6)
            q_ppg = torch.diff(ppg, dim=-1).std(dim=-1)
            q_eda = torch.diff(eda, dim=-1).abs().mean(dim=-1)
            q_tmp = tmp.std(dim=-1)
            q_video = 0.0

            quality_vector = torch.tensor([q_video, q_eeg, q_ppg, q_eda, q_tmp], dtype=torch.float)

            # Safe MinMax Normalization
            min_val = quality_vector.min()
            max_val = quality_vector.max()
            quality_vector = (quality_vector - min_val) / (max_val - min_val + 1e-6)

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