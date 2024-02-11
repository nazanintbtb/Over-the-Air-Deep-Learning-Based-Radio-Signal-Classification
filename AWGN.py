import numpy as np

class AWGN:
    def __init__(self, SNR_dB, signal_power='measured'):
        self.SNR_dB = SNR_dB
        self.signal_power = signal_power

    def AWGN_noise(self, input_signal):
        if self.signal_power == 'measured':
            self.signal_power = np.mean(np.abs(input_signal) ** 2)
        else:
            self.signal_power = 10 ** (self.signal_power / 10)  # Convert dBW to linear scale

        noise_power = self.signal_power / (10 ** (self.SNR_dB / 10))

        noise_real = np.random.normal(0, np.sqrt(noise_power), len(input_signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power), len(input_signal))

        # Add noise
        noisy_signal = input_signal + noise_real + 1j * noise_imag

        return noisy_signal

