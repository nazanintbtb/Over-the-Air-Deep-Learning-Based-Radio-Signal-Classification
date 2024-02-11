import numpy as np

class RicianChannel:
    def __init__(self, sample_rate=1, path_delays=0, average_path_gains=0, k_factor=3, max_doppler_shift=0.001):
        self.sample_rate = sample_rate
        self.path_delays = path_delays
        self.average_path_gains = average_path_gains
        self.k_factor = k_factor
        self.max_doppler_shift = max_doppler_shift

    def reset(self):
        self.sample_rate = 1
        self.path_delays = 0
        self.average_path_gains = 0
        self.k_factor = 3
        self.max_doppler_shift = 0.001

    def filter_signal(self, input_signal):
        num_samples = len(input_signal)
        num_paths = len(self.path_delays)


        path_gains_linear = 10 ** (np.array(self.average_path_gains) / 10)
        k_factor_linear = 10 ** (self.k_factor / 10)


        fading_channel_response_real = np.zeros(num_samples)
        fading_channel_response_imag = np.zeros(num_samples)
        for i in range(num_paths):
            if(self.average_path_gains[i]!=0):
                doppler_frequency = np.random.uniform(-self.max_doppler_shift, self.max_doppler_shift)
                fading_channel_response_real += path_gains_linear[i] * np.cos(2 * np.pi * doppler_frequency * np.arange(num_samples))
                fading_channel_response_imag += path_gains_linear[i] * np.sin(2 * np.pi * doppler_frequency * np.arange(num_samples))

        # Apply fading to the input
        fading_signal_real = input_signal.real * fading_channel_response_real - input_signal.imag * fading_channel_response_imag
        fading_signal_imag = input_signal.imag * fading_channel_response_real + input_signal.real * fading_channel_response_imag

        output_signal = fading_signal_real + 1j * fading_signal_imag

        # Apply line-of-sight
        output_signal += np.sqrt(k_factor_linear) * np.cos(2 * np.pi * doppler_frequency * np.arange(num_samples))

        max_input_amplitude = np.max(np.abs(input_signal))
        max_output_amplitude = np.max(np.abs(output_signal))
        output_signal *= max_input_amplitude / max_output_amplitude


        return output_signal

