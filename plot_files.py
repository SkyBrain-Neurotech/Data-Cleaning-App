import mne

filterd_data = 'filtered_ica_eeg_data.fif'

def load_file_plot(file_name):
    file_path = file_name  # Replace with your actual .fif file path
    raw = mne.io.read_raw_fif(file_path, preload=True)

    # Access metadata from the raw.info dictionary
    sfreq = raw.info['sfreq']  # Sampling frequency
    ch_names = raw.info['ch_names']  # Channel names
    subject_info = raw.info.get('subject_info', {})  # Subject information, if available
    meas_date = raw.info['meas_date']  # Measurement date

    # Print some metadata information
    print(f"Sampling Frequency: {sfreq} Hz")
    print(f"Channel Names: {ch_names}")
    if subject_info:
        print(f"Subject Information: {subject_info}")
    print(f"Measurement Date: {meas_date}")

    # Visualize the raw EEG data (optional)
    raw.plot(n_channels=10, scalings='auto', title='Raw EEG Data', show=True, block=True)

load_file_plot(filterd_data)
