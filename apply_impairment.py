import matplotlib.pyplot as plt
from scipy.signal import lfilter
from RicianChannel import RicianChannel
from PhaseFrequencyOffset import PhaseFrequencyOffset
from AWGN import AWGN
import numpy as np

class Core_process:

    def __init__(self, syms,M, fc=902e6, fs=200e3, k_factor=4, max_doppler_shift=4, max_offset=5, SNR_dB=30, rolloff=0.35  ):
        self.syms = syms
        self.M=M
        self.fc=fc
        self.fs=fs
        self.k_factor=k_factor
        self.max_doppler_shift=max_doppler_shift
        self.max_offset = max_offset
        self.SNR_dB=SNR_dB
        self.rolloff = rolloff

    def reset(self):
        self.syms=0
        self.M=0

    def rcosdesign(self,beta, span, sps, shape='sqrt'):
        if span % 2 != 0:
            raise ValueError("Span must be an even number.")
        if shape != 'sqrt' and shape != 'normal':
            raise ValueError("Shape must be either 'sqrt' or 'normal'.")

        N = span * sps

        t = np.arange(-N / 2, N / 2 + 1) / sps
        t[abs(t) < np.finfo(float).eps] = np.finfo(float).eps

        if shape == 'sqrt':
            num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
            den = np.pi * t * (1 - (4 * beta * t) ** 2)
            h = num / den
            h[np.isnan(h)] = 0
        else:
            h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)

        h = h / np.sqrt(np.sum(h ** 2))
        return h

    def upsample(self,arr, num_zeros):
        upsampled_arr = []
        for complex_num in arr:
            upsampled_arr.append(complex_num)
            for _ in range(num_zeros):
                upsampled_arr.append(0)
        upsampled_arr = np.array(upsampled_arr, dtype=arr.dtype)
        return upsampled_arr

    def apply_impairment(self,name):

        path_delays = np.array([0, 1.8, 3.4]) / self.fs
        average_path_gains = np.array([0, -2, -10])

        filterCoeffs=self.rcosdesign(self.rolloff, self.M, 8)

        y=self.upsample(self.syms,8-1)

        if(name=="CPFS" or name=="GFSK"):
            tx=y
        else:
            tx = lfilter(filterCoeffs, 1, y)

        rician_channel = RicianChannel(self.fs, path_delays, average_path_gains, self.k_factor, self.max_doppler_shift)
        output_signal_RicianChannel = rician_channel.filter_signal(tx)


        clock_offset = (np.random.rand() * 2 * self.max_offset) - self.max_offset
        C = 1 + clock_offset / 1e6
        phase_offset = 0
        frequency_offset =-(C-1)* self.fc;


        freq_shifter = PhaseFrequencyOffset(sample_rate=self.fs,phase_offset=phase_offset, frequency_offset=frequency_offset)
        output_signal_PhaseFrequencyOffset= freq_shifter.apply_offset(output_signal_RicianChannel)

        t = np.arange(len(tx)) / self.fs
        newFs = self.fs * C
        tp = np.arange(len(tx)) / newFs

        vq = np.interp(tp,t, output_signal_PhaseFrequencyOffset)
        awgn = AWGN(self.SNR_dB,signal_power=0)
        noisy_signal = awgn.AWGN_noise(vq)

        # plt.figure(figsize=(10, 6))
        # plt.title(name)
        # plt.plot(noisy_signal.imag, label='imag noisy Signal', color='red')
        # plt.plot(noisy_signal.real, label='real noisy Signal', color='blue')
        #
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        return noisy_signal
