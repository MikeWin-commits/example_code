import os
import glob
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.stats import permutation_cluster_test
import numpy as np
import matplotlib.pyplot as plt

# Directories
epochs_dir = 'epochs'
fsaverage_path = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fsaverage_path)
subject = 'fsaverage'
os.environ['SUBJECTS_DIR'] = subjects_dir

# Import all epochs
epochs_files = glob.glob(os.path.join(epochs_dir, '*.fif'))
epochs_list = [mne.read_epochs(file, preload=True) for file in epochs_files]

# Combine all epochs
all_epochs = mne.concatenate_epochs(epochs_list)

# Separate conditions
success_epochs = all_epochs['Success']
failure_epochs = all_epochs['Failure']

# Compute noise covariance
noise_cov = mne.compute_covariance(all_epochs, tmax=0., method='auto', rank=None)

# Set up source space and BEM model
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Create forward solution
info = all_epochs.info
trans = 'fsaverage'  # fsaverage does not require a specific trans file
fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=False, meg=True)

# Create inverse operator
inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)

# Apply inverse solution to epochs
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc_success = apply_inverse_epochs(success_epochs, inverse_operator, lambda2, method='dSPM', pick_ori=None)
stc_failure = apply_inverse_epochs(failure_epochs, inverse_operator, lambda2, method='dSPM', pick_ori=None)

# Convert to numpy arrays for statistical testing
success_data = np.array([stc.data for stc in stc_success])
failure_data = np.array([stc.data for stc in stc_failure])

# Perform statistical testing
t_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
    [success_data, failure_data],
    n_permutations=1000,
    tail=0,
    threshold=None,
    n_jobs=1
)

# Plot results
plt.figure()
plt.plot(t_obs, label='t-values')
plt.axhline(0, color='k', linestyle='--', label='Zero line')
for i, p_val in enumerate(cluster_p_values):
    if p_val < 0.05:
        h = np.mean(clusters[i], axis=0)
        plt.fill_between(np.arange(len(t_obs)), h * t_obs, where=h != 0, color='red', alpha=0.3, label=f'Cluster {i+1}, p={p_val:.3f}')
plt.xlabel('Time (ms)')
plt.ylabel('t-values')
plt.legend()
plt.title('Statistical Comparison: Success vs. Failure')
plt.show()