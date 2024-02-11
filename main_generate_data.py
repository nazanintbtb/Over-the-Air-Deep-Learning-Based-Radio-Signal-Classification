import numpy as np
from modulation_type import Modulation_Type
from apply_impairment import Core_process
import h5py

if __name__ == "__main__":
    np.random.seed(123456)
    M = 0
    Listmod=["PAM4","QAM16","QAM64","BPSK","8BPS","QPS","CPFS","GFSK"]
    P=Modulation_Type()
    labels=[]
    signals=[]
    for i in range(len(Listmod)):
        if(i==0):
            for j in range(200):
                M=4
                d = np.random.randint(0, M, 1024)
                syms = P.pammod(d)
                c=Core_process(syms,M)
                noisy_signal=c.apply_impairment("PAM4")

                for k in range(0,len(noisy_signal),1024):
                    labels.append([1, 0, 0, 0, 0, 0, 0, 0])
                    # signals.append([noisy_signal[k:k+1024].real,noisy_signal[k:k+1024].imag])

                    sig=noisy_signal[k:k+1024]
                    sign=[]
                    for p in range(1024):
                        sign.append([sig[p].real,sig[p].imag])

                    signals.append((sign))


        elif(i==1):
            for j in range(200):
                M = 16
                d = np.random.randint(0, M, 1024)
                syms=P.qammod_16(d)
                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("QAM16")
                for k in range(0,len(noisy_signal),1024):

                    labels.append([0, 1, 0, 0, 0, 0, 0, 0])
                    # signals.append([noisy_signal[k:k+1024].real,noisy_signal[k:k+1024].imag])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))

        elif(i==2):
            for j in range(200):
                M = 64
                d = np.random.randint(0, M, 1024)
                syms=P.qammod_64(d)
                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("QAM64")
                for k in range(0,len(noisy_signal),1024):

                    labels.append([0, 0, 1, 0, 0, 0, 0, 0])
                    # signals.append(noisy_signal[k:k+1024])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))

        elif(i==3):
            for j in range(200):
                M = 2
                d = np.random.randint(0, M, 1024)
                syms=P.bpskmod(d)
                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("BPSK")

                for k in range(0,len(noisy_signal),1024):
                    labels.append([0, 0, 0, 1, 0, 0, 0, 0])
                    # signals.append(noisy_signal[k:k+1024])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))
        elif(i==4):
            for j in range(200):
                M = 8
                d = np.random.randint(0, M, 1024)
                syms=P.psk8mod(d)
                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("8PSk")
                for k in range(0,len(noisy_signal),1024):
                    labels.append([0, 0, 0, 0, 1, 0, 0, 0])
                    # signals.append(noisy_signal[k:k+1024])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))

        elif(i==5):
            for j in range(200):
                M = 4
                d = np.random.randint(0, M, 1024)
                syms=P.qpskmod(d)
                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("QPS")
                for k in range(0,len(noisy_signal),1024):
                    labels.append([0, 0, 0, 0, 0, 1, 0, 0])
                    # signals.append(noisy_signal[k:k+1024])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))

        elif(i==6):
            for j in range(200):
                M = 2
                d = np.random.randint(0, M, 1024)
                syms=P.cpfskmod(d)

                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("CPFS")
                for k in range(0,len(noisy_signal),1024):
                    labels.append([0, 0, 0, 0, 0, 0, 1, 0])
                    # signals.append(noisy_signal[k:k+1024])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))


        elif(i==7):
            for j in range(200):
                M = 2
                d = np.random.randint(0, M, 1024)
                syms=P.gfskmod(d)
                # M = len(set(syms))
                # if M % 2 != 0:
                #     M += 1
                c = Core_process(syms, M)
                noisy_signal = c.apply_impairment("GFSK")
                for k in range(0,len(noisy_signal),1024):
                    labels.append([0, 0, 0, 0, 0, 0, 0, 1])
                    # signals.append(noisy_signal[k:k+1024])
                    sig = noisy_signal[k:k + 1024]
                    sign = []
                    for p in range(1024):
                        sign.append([sig[p].real, sig[p].imag])
                    signals.append((sign))



    labels=np.array(labels)
    signals=np.array(signals)
    # print(signals[0])
    print(signals.shape)
    print(labels.shape)
    with h5py.File('dataset.h5', 'w') as hf:

        hf.create_dataset('signals', data=signals)
        hf.create_dataset('labels', data=labels)