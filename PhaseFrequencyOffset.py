import numpy as np

class PhaseFrequencyOffset:
    def __init__(self, sample_rate=1, phase_offset=0, frequency_offset=0):
        self.sample_rate = sample_rate
        self.phase_offset = phase_offset
        self.frequency_offset = frequency_offset

    def apply_offset(self, input_signal):
        num_samples = len(input_signal)
        time = np.arange(num_samples) / self.sample_rate

        # Apply phase offset
        phase_radians = np.deg2rad(self.phase_offset)
        phase_shifted_signal = input_signal * np.exp(1j * phase_radians)

        # Apply frequency offset
        frequency_shifted_signal = phase_shifted_signal * np.exp(1j * 2 * np.pi * self.frequency_offset * time)

        return frequency_shifted_signal