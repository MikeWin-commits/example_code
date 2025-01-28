from time import perf_counter
from pyprep.find_noisy_channels import NoisyChannels
import mne
from autoreject import AutoReject
import matplotlib.pyplot as plt
import numpy as np
import os

# Directory paths
data_dir = 'randomflick_nep'
output_dir = 'motor_nep'
os.makedirs(output_dir, exist_ok=True)

# Process all .bdf files in the directory
for file in os.listdir(data_dir):
    if file.endswith('.bdf'):
        file_path = os.path.join(data_dir, file)
        subject_name = os.path.splitext(file)[0]

        print('##################################')
        print("Processing file:", file)
        print('##################################')

        raw = mne.io.read_raw_bdf(file_path, preload=True)

        # Set electrode locations
        raw.drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
        montage = 'biosemi64'
        raw.set_montage(montage)

        # Downsample and Filter
        current_sfreq = raw.info['sfreq']
        desired_sfreq = 200  # Hz
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        lowpass_freq = obtained_sfreq / 3

        raw_downsampled = raw.copy().filter(l_freq=1, h_freq=lowpass_freq)

        # Initialize RANSAC
        nd = NoisyChannels(raw_downsampled)
        nd.find_bad_by_ransac(channel_wise=True)
        raw_downsampled.info['bads'] = nd.bad_by_ransac
        raw_interp = raw_downsampled.copy().interpolate_bads()

        # Find events and set epochs
        events = mne.find_events(raw_downsampled, stim_channel=None, output='onset', consecutive='increasing',
                                 min_duration=0, shortest_event=2, mask=None, uint_cast=False, mask_type='and',
                                 initial_event=False, verbose=None)

        event_mapping = {'Recording Onset': 111, 'Stimulus Onset': 96, 'Success': 98, 'Trial End': 100, 'Failure': 101,
                         'Recording End': 112}

        # Initialize Epochs
        tmin, tmax = -0.8, 0.2
        baseline = (None, 0)  
        picks = mne.pick_types(raw_interp.info, eeg=True, stim=True)
        epochs = mne.Epochs(raw_interp, events, event_mapping, tmin, tmax,
                            picks=picks, baseline=None, reject=None, decim=decim,
                            preload=True)

        print(
            "desired sampling frequency was {} Hz; decim factor of {} yielded an "
            "actual sampling frequency of {} Hz.".format(
                desired_sfreq, decim, epochs.info["sfreq"]
            )
        )

        # Run AUTOREJECT
        ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True)
        ar.fit(epochs)
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)

        # Fit ICA
        ica = mne.preprocessing.ICA(random_state=99)
        ica.fit(epochs[~reject_log.bad_epochs])
        eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=['Fp1', 'F8'], threshold=1.96)
        ica.exclude = eog_indices
        ica.apply(epochs, exclude=ica.exclude)

        # Get rejection threshold and drop bad epochs
        from autoreject import get_rejection_threshold  # noqa
        reject = get_rejection_threshold(epochs, decim=1)
        epochs.drop_bad(reject=reject)

        # Run and fit AUTOREJECT again
        ar.fit(epochs)
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)

        # Create average reference projection
        epochs_avgref = epochs_ar.set_eeg_reference('average', projection=True).resample(sfreq=current_sfreq)
        epochs_refilt = epochs_avgref.copy().filter(1, 30)

        # Save processed data
        avg_epo_path = os.path.join(output_dir, f'{subject_name}-avg-epo.fif')
        epochs_refilt.save(avg_epo_path, overwrite=True)

        # Make Evokeds
        conditions = ['Stimulus Onset', 'Success', 'Failure']
        evokeds = {c: epochs_refilt[c].average() for c in conditions}
        evoked_path = os.path.join(output_dir, f'{subject_name}-ave.fif')
        mne.write_evokeds(evoked_path, list(evokeds.values()), overwrite=True)

        print(f"Finished processing {file}\n")
