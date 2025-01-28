import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from mne import Epochs, create_info
from mne.io import read_raw_fif
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import mne 

# Directories
data_dir = 'raw'
output_dir = 'processed'
os.makedirs(output_dir, exist_ok=True)

# Load all raw FIF files
data_files = glob.glob(os.path.join(data_dir, '*_raw.fif'))

for file_path in data_files:
    subject_name = os.path.basename(file_path).split('_')[0]
    print(f'Processing file: {file_path}')

    raw = read_raw_fif(file_path, preload=True)

    print("Channels:", raw.info["ch_names"])
    raw.load_data()

    # Picking EEG channels
    picks = mne.pick_types(raw.info, eeg=True)

    # Event parameters
    tmin, tmax = -0.2, 1
    event_ids = dict(Stim=96, FC=100)
    events = mne.find_events(
        raw,
        stim_channel=None,
        output='onset',
        consecutive='increasing',
        min_duration=0,
        shortest_event=2
    )

    epochs = Epochs(
        raw,
        events,
        event_ids,
        tmin,
        tmax,
        baseline=None,
        preload=True,
    )
    epochs.drop_channels('Status')
    epochs.resample(200, npad="auto").filter(1, 30)

    # Classification parameters
    clf = make_pipeline(
        CSP(n_components=4, reg=None, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )
    n_splits = 3  # For cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Frequency parameters
    tmin, tmax = -0.200, 1.000
    n_cycles = 10.0
    min_freq = 8.0
    max_freq = 20.0
    n_freqs = 6
    freqs = np.linspace(min_freq, max_freq, n_freqs)
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))

    window_spacing = n_cycles / np.max(freqs) / 2.0
    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    n_windows = len(centered_w_times)

    le = LabelEncoder()
    freq_scores = np.zeros((n_freqs - 1,))

    for freq, (fmin, fmax) in enumerate(freq_ranges):
        w_size = n_cycles / ((fmax + fmin) / 2.0)
        raw_filter = raw.copy().filter(fmin, fmax, fir_design="firwin", skip_by_annotation="edge")

        epochs = Epochs(
            raw_filter,
            events,
            event_ids,
            tmin - w_size,
            tmax + w_size,
            proj=False,
            baseline=None,
            preload=True,
        )
        epochs.drop_bad()
        y = le.fit_transform(epochs.events[:, 2])
        X = epochs.get_data()

        freq_scores[freq] = np.mean(
            cross_val_score(estimator=clf, X=X, y=y, scoring="roc_auc", cv=cv), axis=0
        )

    plt.bar(
        freqs[:-1], freq_scores, width=np.diff(freqs)[0], align="edge", edgecolor="black"
    )
    plt.xticks(freqs)
    plt.ylim([0, 1])
    plt.axhline(
        len(epochs["Stim"]) / len(epochs), color="k", linestyle="--", label="chance level"
    )
    plt.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Decoding Scores")
    plt.title(f"Frequency Decoding Scores - {subject_name}")
    plt.savefig(os.path.join(output_dir, f'{subject_name}_frequency_decoding.png'))
    plt.close()

    tf_scores = np.zeros((n_freqs - 1, n_windows))

    for freq, (fmin, fmax) in enumerate(freq_ranges):
        w_size = n_cycles / ((fmax + fmin) / 2.0)
        raw_filter = raw.copy().filter(fmin, fmax, fir_design="firwin", skip_by_annotation="edge")

        epochs = Epochs(
            raw_filter,
            events,
            event_ids,
            tmin - w_size,
            tmax + w_size,
            proj=False,
            baseline=None,
            preload=True,
        )
        epochs.drop_bad()
        y = le.fit_transform(epochs.events[:, 2])

        for t, w_time in enumerate(centered_w_times):
            w_tmin = w_time - w_size / 2.0
            w_tmax = w_time + w_size / 2.0
            X = epochs.copy().crop(w_tmin, w_tmax).get_data()

            tf_scores[freq, t] = np.mean(
                cross_val_score(estimator=clf, X=X, y=y, scoring="roc_auc", cv=cv), axis=0
            )

    av_tfr = AverageTFR(
        create_info(["freq"], raw.info["sfreq"]),
        tf_scores[np.newaxis, :],
        centered_w_times,
        freqs[1:],
        1,
    )
    chance = np.mean(y)
    av_tfr.plot([0], vmin=chance, title=f"Time-Frequency Decoding Scores - {subject_name}", cmap=plt.cm.Reds)
    plt.savefig(os.path.join(output_dir, f'{subject_name}_time_frequency_decoding.png'))
    plt.close()

    print(f"Finished processing {file_path}\n")