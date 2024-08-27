import streamlit as st
import pandas as pd
import numpy as np
import mne
import io
import os
import tempfile
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# Function to load EEG data into MNE RawArray
def load_eeg_data(data, channel_names, sfreq):
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data.T, info)
    return raw

# Function to clean EEG data by replacing inf and NaN values
def clean_eeg_data(raw):
    data = raw.get_data()
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN, +inf, -inf with 0.0
    raw._data = data
    return raw

# Function to apply bandpass filter
def apply_bandpass_filter(raw, l_freq, h_freq):
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    return raw_filtered

# Function to apply ICA
def apply_ica(raw_filtered, n_components):
    ica = ICA(n_components=n_components, random_state=97, max_iter=800)
    ica.fit(raw_filtered)
    return ica

# Function to apply ICA and remove artifacts
def apply_ica_and_remove_artifacts(raw_filtered, ica):
    raw_cleaned = raw_filtered.copy()
    ica.apply(raw_cleaned)
    return raw_cleaned

# Function to save the processed data as a CSV file in memory
def save_as_csv(raw, channel_names):
    data = raw.get_data().T  # Transpose to get channels as columns
    df = pd.DataFrame(data, columns=channel_names)
    return df.to_csv(index=False).encode('utf-8')

# Function to save the processed data as a FIF file in memory
def save_as_fif(raw):
    with io.BytesIO() as buffer:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fif') as temp_fif:
            raw.save(temp_fif.name, overwrite=True)
            temp_fif.seek(0)
            buffer.write(temp_fif.read())
        os.remove(temp_fif.name)  # Clean up the temporary file
        return buffer.getvalue()

# Function to plot time series of raw, filtered, and ICA-cleaned data
def plot_time_series(raw, filtered, cleaned, channel_names):
    st.write("Time Series: Raw vs Filtered vs Filtered + ICA")
    fig, ax = plt.subplots(len(channel_names), 1, figsize=(15, 2 * len(channel_names)))

    for i, channel in enumerate(channel_names):
        time = raw.times
        ax[i].plot(time, raw.get_data(picks=channel).T, label='Raw Signal', alpha=0.6)
        ax[i].plot(time, filtered.get_data(picks=channel).T, label='Filtered Signal', alpha=0.6)
        ax[i].plot(time, cleaned.get_data(picks=channel).T, label='Filtered + ICA Signal', alpha=0.6)
        ax[i].set_title(f'Channel: {channel}')
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('Amplitude')
        ax[i].legend(loc='upper right')

    fig.tight_layout()
    st.pyplot(fig)

# Function to plot PSD for all channels
def plot_psd_all_channels(raw, filtered, cleaned, channel_names):
    st.write("Power Spectral Density (PSD) Comparison for All Channels")
    
    # Create a figure with three subplots: Raw, Bandpass, and Bandpass + ICA
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot PSD for raw signal
    raw.plot_psd(picks=channel_names, ax=axes[0], fmax=100, show=False, color='blue', average=True, n_fft=2048)
    axes[0].set_title("Raw Signal PSD")
    
    # Plot PSD for bandpass-filtered signal
    filtered.plot_psd(picks=channel_names, ax=axes[1], fmax=100, show=False, color='orange', average=True, n_fft=2048)
    axes[1].set_title("Bandpass-Filtered Signal PSD")
    
    # Plot PSD for bandpass-filtered + ICA cleaned signal
    cleaned.plot_psd(picks=channel_names, ax=axes[2], fmax=100, show=False, color='green', average=True, n_fft=2048)
    axes[2].set_title("Bandpass-Filtered + ICA Cleaned Signal PSD")
    
    # Set common labels
    for ax in axes:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB/Hz)')
    
    fig.tight_layout()
    st.pyplot(fig)

# Streamlit app for EEG data processing and comparison
def main():
    # Set background color
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #150029;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add company logo and name to the dashboard
    st.image("/home/ubuntu/Data-Cleaning-App/new_logo.png", width=400)
    st.title("SkyBrain: EEG Data Processing and Analysis")

    st.write("""
    ### Upload your EEG data in CSV format
    Use this interface to upload your EEG data, process it, and download the processed data. 
    You can apply bandpass filtering, ICA, and save the final results.
    """)

    uploaded_file = st.file_uploader("Upload your EEG data (CSV file)", type=["csv"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        eeg_data = pd.read_csv(uploaded_file)

        channel_names = list(eeg_data.columns)
        fs = 500  # Sampling frequency

        # Convert EEG data to MNE RawArray
        raw = load_eeg_data(eeg_data.values, channel_names, sfreq=fs)

        # Clean the data to remove infs and NaNs
        raw = clean_eeg_data(raw)

        # Apply Bandpass Filter + ICA on all channels
        l_freq = st.slider('Low Frequency (Hz)', 0.1, 50.0, 1.0)
        h_freq = st.slider('High Frequency (Hz)', 50.0, 100.0, 40.0)
        n_components = st.slider('Number of ICA Components', 1, len(channel_names), len(channel_names))

        if st.button("Clean and Process EEG Data"):
            # Apply bandpass filter
            filtered = apply_bandpass_filter(raw, l_freq, h_freq)
            # Apply ICA
            ica = apply_ica(filtered, n_components)
            cleaned = apply_ica_and_remove_artifacts(filtered, ica)
            st.session_state['filtered'] = filtered
            st.session_state['cleaned'] = cleaned

        # Buttons to plot Time Series and PSD
        if 'filtered' in st.session_state and 'cleaned' in st.session_state:
            if st.button("Plot Time Series"):
                plot_time_series(raw, st.session_state['filtered'], st.session_state['cleaned'], channel_names)

            if st.button("Plot PSD"):
                plot_psd_all_channels(raw, st.session_state['filtered'], st.session_state['cleaned'], channel_names)

        # Save processed data as CSV in memory
        if 'cleaned' in st.session_state:
            csv_data = save_as_csv(st.session_state['cleaned'], channel_names)

            # Provide a download button for the CSV
            st.download_button(
                label="Download Filtered + ICA EEG Data as CSV",
                data=csv_data,
                file_name="filtered_ica_eeg_data.csv",
                mime="text/csv"
            )

            # Save processed data as FIF in memory
            fif_data = save_as_fif(st.session_state['cleaned'])

            # Provide a download button for the FIF
            st.download_button(
                label="Download Filtered + ICA EEG Data as FIF",
                data=fif_data,
                file_name="filtered_ica_eeg_data.fif",
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
