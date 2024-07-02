"""Mel feature computation functions."""

import numpy as np
import librosa

def log_mel_spectrogram(data,
                        audio_sample_rate=16000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mel_bins=64,
                        lower_edge_hertz=125.0,
                        upper_edge_hertz=7500.0):
    """Convert waveform to a log-mel-spectrogram.

    Args:
        data: 1-D np.array of waveform data.
        audio_sample_rate: The sampling rate of data.
        log_offset: Add this to values when taking log to avoid -Infs.
        window_length_secs: Duration of each window to analyze.
        hop_length_secs: Step between successive windows.
        num_mel_bins: Number of mel-frequency bins.
        lower_edge_hertz: Lower bound on the frequencies to be included in the mel spectrum.
        upper_edge_hertz: The frequency upper bound.

    Returns:
        2-D np.array of log-mel-spectrogram (shape: [num_frames, num_mel_bins])
    """
    # Compute spectrogram using librosa
    spectrogram = librosa.feature.melspectrogram(
        y=data,
        sr=audio_sample_rate,
        n_fft=int(window_length_secs * audio_sample_rate),
        hop_length=int(hop_length_secs * audio_sample_rate),
        n_mels=num_mel_bins,
        fmin=lower_edge_hertz,
        fmax=upper_edge_hertz)
    
    # Convert to log scale
    log_mel_spectrogram = np.log(spectrogram + log_offset)
    
    return log_mel_spectrogram

def frame(data, window_length, hop_length):
    """Convert spectrogram to overlapping frames.

    Args:
        data: 2-D np.array of spectrogram (shape: [num_frames, num_bins]).
        window_length: Number of frames in each example window.
        hop_length: Number of frames to step forward between examples.

    Returns:
        3-D np.array of framed spectrogram (shape: [num_examples, window_length, num_bins])
    """
    num_samples = data.shape[0]
    num_features = data.shape[1]
    
    num_frames = 1 + int((num_samples - window_length) / hop_length)
    
    shape = (num_frames, window_length, num_features)
    strides = (data.strides[0] * hop_length, data.strides[0], data.strides[1])
    
    framed_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    return framed_data
