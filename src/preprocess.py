import argparse
from glob import glob
import gzip
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend
from typing import List
import tempfile
import subprocess
import h5py
import time




def generate_normalized_names(channel_names):
    """
    Generate a dictionary to map original EEG channel names to their normalized
    names based on the standard 10-20 system naming convention, ensuring
    proper case (e.g., 'Fp1' instead of 'FP1').
    Parameters:
    - channel_names: list of str, original channel names from the EEG file.
    Returns:
    - dict: A dictionary where keys are original channel names and values are
            the corresponding normalized names in proper case.
    """
    prefix_removal = "EEG "
    suffix_removals = ["-REF", "-LE"]

    # Mapping of upper case to proper case for the 10-20 system
    proper_case_mapping = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'F7': 'F7', 'F3': 'F3', 'FZ': 'Fz', 'F4': 'F4', 'F8': 'F8',
        'T1': 'T1', 'T3': 'T3', 'C3': 'C3', 'CZ': 'Cz', 'C4': 'C4', 'T4': 'T4', 'T2': 'T2',
        'T5': 'T5', 'P3': 'P3', 'PZ': 'Pz', 'P4': 'P4', 'T6': 'T6',
        'O1': 'O1', 'OZ': 'Oz', 'O2': 'O2',
    }

    normalized_names = {}
    for name in channel_names:
        name_short = name
        # Remove the 'EEG ' prefix and any '-REF' or '-LE' suffix
        for suffix in suffix_removals:
            name_short = name_short.replace(suffix, '')
        name_short = name_short.replace(prefix_removal, '')

        # Convert to proper case based on the 10-20 system
        normalized_name = proper_case_mapping.get(name_short.upper())

        # Map the original name to the normalized name
        normalized_names[name] = normalized_name

    return normalized_names

def preprocess_eeg(edf_file, logger, channels = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']):
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    # raw = mne.io.read_raw_edf(edf_file, preload=False, verbose='error')

    # Select the 22 channels based on the extended international 10-20 system
    ch_suffixes = [ch.split("-")[-1] for ch in raw.info['ch_names'] if ch.endswith("LE") or ch.endswith("REF")]
    assert len(set(ch_suffixes))==1, f"Multiple channel type detected: {set(ch_suffixes)}"
    channels_formatted = [f'EEG {ch.upper()}-{ch_suffixes[0]}' for ch in channels]
    missing_channels = list(set(channels_formatted)-set(raw.info['ch_names']))
    if len(missing_channels):
        logger.info(f"Missing channels {missing_channels} in available channels: \n{raw.info['ch_names']}")

    if missing_channels:
        logger.info(f"Available channels: \n{raw.info['ch_names']}\nAdding missing channels {missing_channels} with zero values...")
        for ch_name in missing_channels:
            # Create a data array of zeros
            data = np.zeros((1, len(raw.times)))
            # Create an Info object for the new channel
            ch_info = mne.create_info(ch_names=[ch_name], sfreq=raw.info['sfreq'], ch_types='eeg')
            # Create a RawArray and append to the existing Raw object
            missing_raw = mne.io.RawArray(data, ch_info)
            raw.add_channels([missing_raw], force_update_info=True)

        # Mark the newly added channels as bad
        raw.info['bads'].extend(missing_channels)

    # Ensure the specified channels are in the correct order
    raw.reorder_channels(channels_formatted)

    logger.info("Selecting the 22 channels...")
    raw.pick_channels(channels_formatted, ordered=False)

    # Identify bad channels (zero or missing signals)
    logger.info("Identifying bad channels...")
    bad_channels = []
    for ch_name in channels_formatted:
        data, _ = raw[ch_name]
        if np.all(data == 0):
            bad_channels.append(ch_name)

    raw.info['bads'] = bad_channels

    # Rename channels in the raw object
    normalized_names = generate_normalized_names(raw.info['ch_names'])
    raw.rename_channels(normalized_names)
    # Set montage (assuming 10-20 system)
    montage = mne.channels.make_standard_montage('standard_1020')
    # remove channels not present in the 10-20 system
    drop_channels = [ch for ch in raw.info['ch_names'] if ch not in montage.ch_names]
    logger.info(f"Dropping channels: {drop_channels}")

    raw.drop_channels(drop_channels)
    raw.set_montage(montage, match_case=False)
    # Interpolate bad channels (This is a simplified approach)
    logger.info("Interpolating bad channels...")
    if bad_channels:
        logger.info(f"Processing bad channels: {bad_channels}")
        raw.interpolate_bads(reset_bads=True)

    logger.info("Processing all the channels...")
    # Re-reference the EEG signal to the average
    raw.set_eeg_reference(ref_channels='average')

    # Remove power line noise with notch filter and apply bandpass filter
    raw.notch_filter(60, notch_widths=1)
    raw.filter(0.5, 100, fir_design='firwin')

    # Resample to 250 Hz
    raw.resample(250)

    # DC offset correction and remove linear trends
    data = raw.get_data()
    data = detrend(data, type='constant')  # DC offset correction
    data = detrend(data, type='linear')    # Remove linear trends

    # Apply the z-transform along the time dimension
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    raw._data = data

    return raw

def process_file(edf_file, export_dir, new_filename, logger, format, delete=False):
    logger.info(f"Processing {edf_file}")
    # Base export directory
    export_dir = Path(export_dir)
    # define the new filename
    if os.path.isfile(export_dir / new_filename):
        logger.info(f"File already processed. Skipping {new_filename}")
        return
    try:
        with preprocess_eeg(str(edf_file), logger) as preprocessed:
            # Full path for the preprocessed file
            save_preprocessed_data(preprocessed, export_dir / f"{new_filename}", logger, format=format)
        if delete:
            os.remove(edf_file)
    except Exception as e:
        # raise e
        logger.error(f"Error processing {edf_file}: {e}")

# def process_subject(subject_path, filenames_to_process, file_prefix):
#     for edf_file in subject_path.rglob('*.edf'):
#         preprocessed_file_name = f"{edf_file.stem}_preprocessed.pt"
#         if preprocessed_file_name in filenames_to_process:
#             logger.info(f"Original file exists: {preprocessed_file_name}.")
#             process_file(edf_file, file_prefix)
#         else:
#             print(f"Original file does not exist: {preprocessed_file_name}.")
def save_to_hdf5(raw, output_path, logger):
    data = raw.get_data()
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('eeg_data', data=data, compression="gzip")
        f.attrs['sfreq'] = raw.info['sfreq']  # Save metadata
        f.attrs['ch_names'] = ','.join(raw.info['ch_names'])
        f.attrs['times'] = raw.times
    logger.info(f"Saved HDF5 file: {output_path}")

def save_to_npy(raw, output_path, logger):
    np.save(output_path, raw.get_data())
    logger.info(f"Saved NumPy file: {output_path}")

def save_preprocessed_data(raw, output_path, logger, format='npy'):
    if format == 'h5':
        save_to_hdf5(raw, output_path, logger)
    elif format == 'npy':
        save_to_npy(raw, output_path, logger)
    # elif format == 'tfrecord':
    #     save_to_tfrecord(raw, output_path)
    # elif format == 'parquet':
    #     save_to_parquet(raw, output_path)
    # elif format == 'wds':
    #     save_to_webdataset(raw, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def process_and_save(args=None, data_root=None, export_dir=None, logger=None, delete=False):
    if args:
        data_root, export_dir = args.data_root, args.export_dir

    os.makedirs(export_dir, exist_ok=True)
    # filename format: tuh/tueg/edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf
    # processed filename format: 000_01_tcp_ar_aaaaaaaa_s001_t000_preprocessed.pt
    data_root_path = Path(data_root)
    export_root_path = Path(export_dir)

    files_processed = 0
    start_time = time.time()
    for div in os.listdir(data_root_path):
        div_path = os.path.join(data_root_path, div)
        for subject_num in os.listdir(div_path):
            subject_path = os.path.join(div_path, subject_num)
            for session in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session)
                for montage in os.listdir(session_path):
                    montage_path = os.path.join(session_path, montage)
                    for file in os.listdir(montage_path):
                        print(montage_path, file)
                        if file:
                            edf_file = os.path.join(montage_path, file)
                            token = file[-8:-4]
                            file_pref = f"{subject_num}_{session}_{token}_{montage}"
                            file_suf = ".npy"
                            new_file_nm = file_pref + file_suf
                            print(f"edf_file_to_process: {edf_file}")
                            process_file(edf_file=edf_file, export_dir=export_dir, new_filename=new_file_nm,
                                         logger=logger, format='npy', delete=False)
                            # process_file(edf_file, export_dir, new_file_nm, logger, delete)
                            files_processed += 1
    end_time = time.time()
    time_elapsed = end_time - start_time
    logger.info(f"{files_processed} processsed in {time_elapsed} seconds.")
    logger.info(f"Average time to process: {(time_elapsed / files_processed):.3}s per file.")

        # subject_path = data_root_path / subject_num
        # # Adapted pattern to match 'sNNN_YYYY'
        # for session_dir in subject_path.rglob('s*_*'):
        #     if session_dir.is_dir():
        #         logger.info(f"Processing {session_dir}...")
        #         logger.info(f"subject num: {subject_num}")
        #         logger.info(f"session dir: {session_dir}")
        #         child_dirs = glob(str(session_dir)+'//*')
        #         assert len(child_dirs)==1, f"Multiple child dirs found: {child_dirs}"
        #         session_type = child_dirs[0].split('/')[-1]
        #         file_prefix = f"{subject_num}_{session_type}"
        #         process_subject(session_dir, export_root_path, filenames_to_process, file_prefix)




if __name__ == "__main__":
    ...
    # edfs = []
    # for dir_path, dirs, files in os.walk("../data/np"):
    #     for file in files:
    #         if os.path.isfile(Path(dir_path) / file):
    #             data = np.load(Path(dir_path) / file)
    #             print(data.shape)






    # parser = argparse.ArgumentParser(description="Preprocess EEG data.")
    # parser.add_argument("--data-root", required=True, help="Root directory of the EEG data.")
    # parser.add_argument("--export-dir", required=True, help="Directory where the preprocessed data will be saved.")
    # parser.add_argument("--filename-csv", default="../inputs/sub_list2.csv", help="CSV file containing the list of filenames to process.")
    #
    # Configure logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    # process_and_save(data_root="../data/edf", export_dir="../data/np", logger=logger)

    # data_root= "../data/edf"
    # export_dir = "../data/pt"
    # # new_name = f"aaaaaaaa_s001_2015_{eeg_file[-8:-4]}_01_tcp_ar.pt"
    # process_and_save(data_root=data_root, export_dir=export_dir, logger=logger)
    #
    # args = parser.parse_args()
    #
    # process_and_save(args)
