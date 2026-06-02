import torch
import numpy as np
import pickle
import cv2


class MultimodalDataset:

    def __init__(self, subject_list, data_path, video_path, window_size=5, fs=256):

        self.subject_list = subject_list
        self.data_path = data_path
        self.video_path = video_path

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

            eeg = d["data"][:, 0:32, :]        # [trials, channels, time]
            ppg, eda, tmp = d["data"][:, 37:40, :]      # adjust if needed
            labels = d["labels"]

            num_trials = eeg.shape[0]

            for t in range(num_trials):

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
    def load_video_window(self, subject, trial, start):

        path = f"{self.video_path}/{subject}/{trial}.avi"

        cap = cv2.VideoCapture(path)

        frames = []
        idx = 0

        start_frame = start  # assume synced index (or map via fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx >= start_frame and len(frames) < self.window_samples:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            idx += 1

        cap.release()

        frames = np.stack(frames)
        frames = torch.tensor(frames).permute(0, 3, 1, 2)

        return frames.float() / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]

        subject = s["subject"]
        trial = s["trial"]
        start = s["start"]

        data = getattr(self, f"{subject}_data")

        eeg = data["eeg"][trial, :, start:start+self.window_samples]
        ppg = data["ppg"][trial, :, start:start+self.window_samples]
        eda = data["eda"][trial, :, start:start+self.window_samples]
        tmp = data["tmp"][trial, :, start:start+self.window_samples]

        video = self.load_video_window(subject, trial, start)

        label = data["labels"][trial]

        return {
            "face": video,     # [T, 3, H, W]
            "eeg": eeg,        # [C, T]
            "ppg": ppg,  # [C, T] or [T]
            "eda": eda,  # [C, T] or [T]
            "tmp": tmp,  # [C, T] or [T]
            "targets": label,
            "subject": subject
        }