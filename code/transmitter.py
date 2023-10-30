import numpy as np


sampling_rate = 1  # sample / second; (is normalized)
sampling_period = 4  # seconds
information_rate = 1 / sampling_period  # symbols / seconds

beta_sqrt_raised_cosine = 0.5


# Step 1: make a function that describes the DTFT for the sqrt raised cosine pulse
def GetSqrtRaisedCosineDTFT(w):
    H = np.zeros(len(w))

    for i in range(len(w)):
        if abs(w[i]) <= (np.pi * (1 - beta_sqrt_raised_cosine)) / sampling_period:
            H[i] = np.sqrt(sampling_period)

        elif abs(w[i]) <= (np.pi * (1 + beta_sqrt_raised_cosine)) / sampling_period:
            H[i] = np.sqrt(
                (sampling_period / 2)
                * (
                    1
                    + np.cos(
                        (sampling_period / (2 * beta_sqrt_raised_cosine))
                        * (
                            np.abs(w[i])
                            - (
                                (np.pi * (1 - beta_sqrt_raised_cosine))
                                / sampling_period
                            )
                        )
                    )
                )
            )

    return H

    # k = np.linspace(0, (n_samples - 1), num=n_samples)


# Step 2: find the sqrt_raised_cosine impulse_response_function given the DFT
#
# h_sqrt_raised_cosine[n] = inverseDFT(H_sqrt_raised_cosine(k))
#                         = inverseDFT(H_sqrt_raised_cosine(e^jw)), where w = (2*pi*k)/N
#                                                                   where k = 0, 1, 2, ... N-1
# NOTE: second half of inverseDFT samples must be shifted all the way to the beginning
def GetSqrtRaisedCosineDFT(n_samples):
    k = np.linspace(
        (n_samples / -2), ((n_samples / 2) - 1), num=n_samples
    )
    w = (2 * np.pi * k) / n_samples
    ifft = np.fft.ifft(GetSqrtRaisedCosineDTFT(w))
    return np.fft.ifftshift(ifft)


# Step 3: truncate the infinite impulse response h so that it can be an FIR filter
def GetFIR():
    filter_sample_count = 32
    return GetSqrtRaisedCosineDFT(filter_sample_count)


# Quick and dirty tests
import matplotlib.pyplot as plt


def TEST_PlotHDTFT():
    test_n_samples = 10000
    k = np.linspace(
        (test_n_samples / -2), ((test_n_samples / 2) - 1), num=test_n_samples
    )
    w = (2 * np.pi * k) / test_n_samples

    test_H = GetSqrtRaisedCosineDTFT(w)

    plt.plot(w, test_H, ".-")
    plt.show()


def TEST_PlotImpulseResponseFunction():
    test_n_samples = 100
    k = np.linspace(
        (test_n_samples / -2), ((test_n_samples / 2) - 1), num=test_n_samples
    )
    test_h = GetSqrtRaisedCosineDFT(test_n_samples)

    plt.plot(k, test_h.real, ".-")
    plt.plot(k, test_h.imag, ".-") # imaginary parts should be zero 
    plt.show()

def TEST_PlotFIR():
    test_n_samples = 32
    k = np.linspace(0, (test_n_samples - 1), num=test_n_samples)
    test_h_fir = GetFIR()

    plt.plot(k, test_h_fir.real, ".:")
    plt.show()


if __name__ == "__main__":
    # TEST_PlotHDTFT()
    # TEST_PlotImpulseResponseFunction()
    TEST_PlotFIR()

