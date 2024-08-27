import streamlit as st
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import io

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

# Function to plot an overview of all channels: raw vs filter vs filter + ICA
def plot_overview_all_channels(raw, filtered, cleaned, channel_names):
    st.write("Overview: Raw vs Filtered vs Filtered + ICA for All Channels")
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(15, 2 * len(channel_names)))

    for i, channel in enumerate(channel_names):
        time = raw.times
        axes[i].plot(time, raw.get_data(picks=channel).T, label='Raw Signal', alpha=0.6)
        axes[i].plot(time, filtered.get_data(picks=channel).T, label='Filtered Signal', alpha=0.6)
        axes[i].plot(time, cleaned.get_data(picks=channel).T, label='Filtered + ICA Signal', alpha=0.6)
        axes[i].set_title(f'Channel: {channel}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend(loc='upper right')

    fig.tight_layout()
    st.pyplot(fig)

# Function to plot time series or PSD for a selected channel
def plot_selected_channel(raw, filtered, cleaned, channel, plot_type):
    if plot_type == "Time Series":
        st.write(f"Time Series Comparison: Channel {channel}")
        fig, ax = plt.subplots(figsize=(12, 6))
        time = raw.times
        ax.plot(time, raw.get_data(picks=channel).T, label='Raw Signal', alpha=0.6)
        ax.plot(time, filtered.get_data(picks=channel).T, label='Filtered Signal', alpha=0.6)
        ax.plot(time, cleaned.get_data(picks=channel).T, label='Filtered + ICA Signal', alpha=0.6)
        ax.set_title(f'Channel: {channel}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        st.pyplot(fig)

    elif plot_type == "PSD":
        st.write(f"Power Spectral Density (PSD) Comparison: Channel {channel}")
        fig, ax = plt.subplots(figsize=(12, 6))
        raw.plot_psd(picks=channel, ax=ax, fmax=100, show=False, color='blue', average=True, n_fft=2048)
        filtered.plot_psd(picks=channel, ax=ax, fmax=100, show=False, color='orange', average=True, n_fft=2048)
        cleaned.plot_psd(picks=channel, ax=ax, fmax=100, show=False, color='green', average=True, n_fft=2048)
        ax.legend(["Raw Signal", "Filtered Signal", "Filtered + ICA Signal"])
        st.pyplot(fig)

# Function to plot PSD comparison for all channels
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

# Function to generate MNE-supported EEG file with metadata
def generate_eeg_file(raw, file_name, extra_metadata=None):
    if extra_metadata:
        # Store custom metadata under 'description' or 'temp' in MNE's Info object
        metadata_str = "; ".join([f"{key}: {value}" for key, value in extra_metadata.items()])
        raw.info['description'] = metadata_str
    raw.save(file_name, overwrite=True)
    st.success(f"EEG data saved as {file_name} with metadata: {extra_metadata}")

# Function to save the processed data as a CSV file in memory
def save_as_csv(raw, channel_names):
    data = raw.get_data().T  # Transpose to get channels as columns
    df = pd.DataFrame(data, columns=channel_names)
    return df.to_csv(index=False)

# Streamlit app for EEG data processing and comparison
def main():
    st.title("EEG Data Processing and Analysis")
    
    st.write("""
    ### Upload your EEG data in CSV format
    Use this interface to upload your EEG data, process it, and download the processed data. 
    You can apply bandpass filtering, ICA, and save the results at each stage.
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

        # Input Metadata for Raw Data
        st.write("### Input Metadata for Raw Data")
        raw_metadata = st.text_area("Enter metadata for raw data (key: value pairs, e.g., Device: ABC, User: John)", "")
        raw_metadata_dict = {}
        if raw_metadata:
            for item in raw_metadata.split('\n'):
                if ':' in item:
                    key, value = item.split(':', 1)  # Split only at the first colon
                    raw_metadata_dict[key.strip()] = value.strip()

        raw_file_name = st.text_input("Enter the file name for the raw EEG data", "raw_eeg_data.fif")
        if st.button("Save Raw Data as .fif"):
            if raw_file_name:
                generate_eeg_file(raw, raw_file_name, raw_metadata_dict)

        # Apply Bandpass Filter on all channels
        l_freq = st.slider('Low Frequency (Hz)', 0.1, 50.0, 1.0)
        h_freq = st.slider('High Frequency (Hz)', 50.0, 100.0, 40.0)
        
        if st.button("Apply Bandpass Filter to All Channels"):
            filtered = apply_bandpass_filter(raw, l_freq, h_freq)
            st.session_state['filtered'] = filtered

            # Save processed data to CSV in memory
            csv_data = save_as_csv(filtered, channel_names)

            # Provide a download button for the CSV
            st.download_button(
                label="Download Filtered EEG Data as CSV",
                data=csv_data,
                file_name="filtered_eeg_data.csv",
                mime="text/csv"
            )

        # Input Metadata for Filtered Data
        if 'filtered' in st.session_state:
            st.write("### Input Metadata for Filtered Data")
            filtered_metadata = st.text_area("Enter metadata for filtered data (key: value pairs, e.g., Device: ABC, User: John)", "")
            filtered_metadata_dict = {}
            if filtered_metadata:
                for item in filtered_metadata.split('\n'):
                    if ':' in item:
                        key, value = item.split(':', 1)  # Split only at the first colon
                        filtered_metadata_dict[key.strip()] = value.strip()

            filtered_file_name = st.text_input("Enter the file name for the filtered EEG data", "filtered_eeg_data.fif")
            if st.button("Save Filtered Data (.fif)"):
                if filtered_file_name:
                    generate_eeg_file(st.session_state['filtered'], filtered_file_name, filtered_metadata_dict)
        
        # Apply Bandpass Filter + ICA on all channels
        n_components = st.slider('Number of ICA Components', 1, len(channel_names), len(channel_names))
        
        if st.button("Apply Bandpass Filter + ICA to All Channels"):
            if 'filtered' not in st.session_state:
                filtered = apply_bandpass_filter(raw, l_freq, h_freq)
            else:
                filtered = st.session_state['filtered']
            
            ica = apply_ica(filtered, n_components)
            cleaned = apply_ica_and_remove_artifacts(filtered, ica)
            st.session_state['cleaned'] = cleaned

            plot_overview_all_channels(raw, filtered, cleaned, channel_names)

            # Save processed data to CSV in memory
            csv_data = save_as_csv(cleaned, channel_names)

            # Provide a download button for the CSV
            st.download_button(
                label="Download Filtered + ICA EEG Data as CSV",
                data=csv_data,
                file_name="filtered_ica_eeg_data.csv",
                mime="text/csv"
            )

        # Input Metadata for Filtered + ICA Data
        if 'cleaned' in st.session_state:
            st.write("### Input Metadata for Filtered + ICA Data")
            cleaned_metadata = st.text_area("Enter metadata for filtered + ICA data (key: value pairs, e.g., Device: ABC, User: John)", "")
            cleaned_metadata_dict = {}
            if cleaned_metadata:
                for item in cleaned_metadata.split('\n'):
                    if ':' in item:
                        key, value = item.split(':', 1)  # Split only at the first colon
                        cleaned_metadata_dict[key.strip()] = value.strip()

            cleaned_file_name = st.text_input("Enter the file name for the filtered + ICA EEG data", "filtered_ica_eeg_data.fif")
            if st.button("Save Filtered + ICA Data (.fif)"):
                if cleaned_file_name:
                    generate_eeg_file(st.session_state['cleaned'], cleaned_file_name, cleaned_metadata_dict)
        
        # Plot PSD for all channels
        if 'cleaned' in st.session_state and st.button("Plot PSD for All Channels"):
            plot_psd_all_channels(raw, st.session_state['filtered'], st.session_state['cleaned'], channel_names)

if __name__ == "__main__":
    main()
