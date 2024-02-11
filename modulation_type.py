import numpy as np

class Modulation_Type:

    def pammod(self,input_data):

       constellation = np.array([-1 - 0j, -3 + 0j, 3 + 0j, 1 - 0j])

       modulated_symbols = constellation[input_data]

       return modulated_symbols


    def qammod_16(self,input_data):
        constellation = np.array([-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
                          -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
                           3 - 3j,  3 - 1j,  3 + 3j,  3 + 1j,
                           1 - 3j,  1 - 1j,  1 + 3j,  1 + 1j])


        modulated_symbols = constellation[input_data]

        return modulated_symbols


    def qammod_64(self,input_data):
        constellation = np.array([-7 - 7j, -7 - 5j, -7 - 3j, -7 - 1j, -7 + 7j, -7 + 5j, -7 + 3j, -7 + 1j,
                                  -5 - 7j, -5 - 5j, -5 - 3j, -5 - 1j, -5 + 7j, -5 + 5j, -5 + 3j, -5 + 1j,
                                  -3 - 7j, -3 - 5j, -3 - 3j, -3 - 1j, -3 + 7j, -3 + 5j, -3 + 3j, -3 + 1j,
                                  -1 - 7j, -1 - 5j, -1 - 3j, -1 - 1j, -1 + 7j, -1 + 5j, -1 + 3j, -1 + 1j,
                                  7 - 7j, 7 - 5j, 7 - 3j, 7 - 1j, 7 + 7j, 7 + 5j, 7 + 3j, 7 + 1j,
                                  5 - 7j, 5 - 5j, 5 - 3j, 5 - 1j, 5 + 7j, 5 + 5j, 5 + 3j, 5 + 1j,
                                  3 - 7j, 3 - 5j, 3 - 3j, 3 - 1j, 3 + 7j, 3 + 5j, 3 + 3j, 3 + 1j,
                                  1 - 7j, 1 - 5j, 1 - 3j, 1 - 1j, 1 + 7j, 1 + 5j, 1 + 3j, 1 + 1j])

        modulated_symbols = constellation[input_data]

        return modulated_symbols

    def qpskmod(self,input_data):

        constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        modulated_symbols = constellation[input_data]
        return modulated_symbols

    def bpskmod(self,input_data):

        constellation = np.array([1+0j, -1+0j])
        modulated_symbols = constellation[input_data]
        return modulated_symbols

    def psk8mod(self,input_data):

        constellation = np.exp(1j * (2 * np.pi / 8) * np.arange(8))
        modulated_symbols = constellation[input_data]
        return modulated_symbols

    def gfskmod(self,input_data, bt=0.5):

        modulated_symbols = np.exp(1j * np.cumsum(input_data) / bt)

        return modulated_symbols

    def cpfskmod(self,input_data, modulation_index=0.5, frequency_deviation=10e3, sampling_rate=200e3):

        phase_increments = 2 * np.pi * frequency_deviation / sampling_rate


        modulated_signal = np.exp(1j * np.cumsum(phase_increments * (input_data - 0.5 * modulation_index)))

        return modulated_signal

    # analog signal modulation
    def bfmmod(self, data, carrier_frequency=10e6, sampling_rate=200e3):

        t = np.linspace(0, 1024 / sampling_rate, 1024)
        modulated_signal = np.sin(2 * np.pi * carrier_frequency * t + data)
        return modulated_signal

    def dsbammod(self, data, carrier_frequency=10e6, sampling_rate=200e3):

        t = np.linspace(0, 1024 / sampling_rate, 1024)
        modulated_signal = data * np.sin(2 * np.pi * carrier_frequency * t)

        return modulated_signal

    def ssbammod(self, data, carrier_frequency=10e6, sampling_rate=200e3, sideband='upper'):
        t = np.linspace(0, 1024 / sampling_rate, 1024)
        if sideband == 'upper':
            modulated_signal = data * np.sin(2 * np.pi * carrier_frequency * t) - np.imag(np.fft.ifft(np.fft.fft(data) * 1j))
        elif sideband == 'lower':
            modulated_signal = data * np.sin(2 * np.pi * carrier_frequency * t) - np.real(np.fft.ifft(np.fft.fft(data) * 1j))
        else:
            raise ValueError("Invalid sideband selection. Please choose 'upper' or 'lower'.")

        return modulated_signal
