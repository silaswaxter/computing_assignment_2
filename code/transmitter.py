import numpy as np
import upsampler
import convolution
from scipy import signal


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


# Step 2: find the sqrt_raised_cosine impulse_response_function given the DFT
#
# h_sqrt_raised_cosine[n] = inverseDFT(H_sqrt_raised_cosine(k))
#                         = inverseDFT(H_sqrt_raised_cosine(e^jw)), where w = (2*pi*k)/N
#                                                                   where k = 0, 1, 2, ... N-1
# NOTE: second half of inverseDFT samples must be shifted all the way to the beginning
def GetSqrtRaisedCosineImpulseResponseFromIFFT(n_samples):
    k = np.linspace(-1 * (n_samples / 2), ((n_samples / 2) - 1), num=n_samples)
    w = (2 * np.pi * k) / n_samples
    # IMPORTANT: in order to take the IDFT, a DFT must be passed. The DFT is related to
    #            the DTFT in this case by a shift. So, in the DTFT the response is from
    #            [(3*pi/8), (-3*pi/8)], but if this was a DFT it'd be from
    #            [-pi, (-pi + 3*pi/8)] and from [(pi - 3*pi/8), pi]. To do this
    #            manipulation `np.fft.fftshift(...)` is used.
    ifft = np.fft.ifft(np.fft.fftshift(GetSqrtRaisedCosineDTFT(w)))
    return np.fft.ifftshift(ifft), w


# Step 3: truncate the infinite impulse response h so that it can be an FIR filter
def GetFIR():
    test_n_samples = 32
    test_h, _ = GetSqrtRaisedCosineImpulseResponseFromIFFT(test_n_samples)
    return test_h.real


# Step 4: shape the input signal using the raised cosine pulse
def ApplyPulseShaping(input_signal):
    # first upsample the input_signal by a factor of 4
    upsampled_input = upsampler.UpSample(input_signal, sampling_period)
    fir = GetFIR()
    pulses_0 = convolution.Convolve_FiniteSequences(upsampled_input[0:, 0], fir)
    pulses_1 = convolution.Convolve_FiniteSequences(upsampled_input[0:, 1], fir)
    pulses = np.array([pulses_0, pulses_1])
    return pulses


# Step 5: upsample and filter
def UpSample20WithLPF(input_signals):
    upsample_n = 20

    upsampled_signals = list()
    for input_signal in input_signals:
        upsampled_signal = upsampler.UpSample(input_signal, upsample_n)
        upsampled_signals.append(upsampled_signal)

    upsampled_signals = np.array(upsampled_signals)

    pre_upsample_cutoff = 3 * np.pi / 8
    cutoff = pre_upsample_cutoff / upsample_n
    # the cutoff values here are calculated for frequencies in terms of rads/sample, but
    # scipy uses 1/sample; correct for that--theres probably a more elegant solution.
    cutoff /= 2 * np.pi
    print(cutoff)
    passband_desired_gain = upsample_n
    stopband_desired_gain = 0
    h_lpf_length = 71
    h = signal.firls(
        h_lpf_length,
        [0, cutoff, cutoff, 0.5],
        [
            passband_desired_gain,
            passband_desired_gain,
            stopband_desired_gain,
            stopband_desired_gain,
        ],
        fs=1,
    )

    filtered_signals = list()
    for upsampled_signal in upsampled_signals:
        filtered_signal = convolution.Convolve_FiniteSequences(upsampled_signal, h)
        filtered_signals.append(filtered_signal)

    filtered_signals = np.array(filtered_signals)
    return filtered_signals, h


def Transmit(input_signal):
    carrier_frequency = 0.44 * np.pi  # radians/sample

    pulses, _ = UpSample20WithLPF(ApplyPulseShaping(input_signal))

    n = np.arange(pulses[0].shape[-1])
    modulated_carrier_signal = pulses[0] * np.cos(carrier_frequency * n) + pulses[
        1
    ] * np.sin(carrier_frequency * n)
    return modulated_carrier_signal


# Quick and dirty tests
import matplotlib.pyplot as plt


# A method to return the random bit sequence so that tests are repeatable.
def TEST_HELPER_GetRandomBitSequence():
    return np.array(
        [
            [-1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [-1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [-1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [-1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    )


def TEST_HELPER_GetRandomBitSequenceTruncated():
    test_data = TEST_HELPER_GetRandomBitSequence()
    upper_index = int(np.floor(len(test_data) / 4))
    return test_data[0:upper_index]


def TEST_PlotHDTFT():
    test_n_samples = 10000
    k = np.linspace(
        (test_n_samples / -2), ((test_n_samples / 2) - 1), num=test_n_samples
    )
    w = (2 * np.pi * k) / test_n_samples

    test_H = GetSqrtRaisedCosineDTFT(w)

    plt.plot(w, test_H, ".-")
    plt.xlabel("Frequency in radians/sample")
    plt.ylabel("Magnitude")
    plt.title("Frequency Response of Root Raised Cosine Filter")
    plt.show()


def TEST_PlotHDFT():
    test_n_samples = 10000
    k = np.linspace(
        (test_n_samples / -2), ((test_n_samples / 2) - 1), num=test_n_samples
    )
    w = (2 * np.pi * k) / test_n_samples

    test_H = np.fft.fftshift(GetSqrtRaisedCosineDTFT(w))

    plt.plot(w, test_H, ".-")
    plt.show()


def TEST_PlotImpulseResponseFunction():
    test_n_samples = 100
    test_h, w = GetSqrtRaisedCosineImpulseResponseFromIFFT(test_n_samples)

    plt.plot(w, test_h.real, ".-")
    plt.plot(w, test_h.imag, ".-")  # imaginary parts should be zero
    plt.show()


def TEST_PlotFIR():
    test_h_fir = GetFIR()

    plt.plot(np.arange(len(test_h_fir)), test_h_fir, ".:")
    plt.xlabel("n")
    plt.ylabel("h[n]")
    plt.title("Impulse Response Sequence of Root Raised Cosine Filter")
    plt.show()


def TEST_ApplyPulseShaping():
    pulses = ApplyPulseShaping(TEST_HELPER_GetRandomBitSequenceTruncated())

    k = np.linspace(0, len(pulses[0]), len(pulses[0]))
    fig, axis = plt.subplots(2)
    axis[0].plot(k, pulses[0], ".:")
    axis[0].set_title("b_1[n] Pulse-Shaped vs. n")
    axis[1].plot(k, pulses[1], ".:")
    axis[1].set_title("b_2[n] Pulse-Shaped vs. n")

    fig.supxlabel("n")
    plt.show()


def TEST_ApplyPulseShapingWithUpSampling():
    pulses = ApplyPulseShaping(TEST_HELPER_GetRandomBitSequence())
    # PLOT NORMAL FFT
    fig, axis = plt.subplots(2)
    norm_fft = 20 * np.log10(abs(np.fft.fft(pulses[0])))
    freqs = np.fft.fftfreq(pulses[0].shape[-1])
    w = (2 * np.pi) * freqs
    axis[0].semilogy(w, norm_fft, ".:")
    axis[0].set_title("b_1 Frequency Response")

    upsample_n = 20
    pulses_up_0 = upsampler.UpSample(pulses[0], upsample_n)
    pulses_up_1 = upsampler.UpSample(pulses[1], upsample_n)
    pulses = np.array([pulses_up_0, pulses_up_1])

    # PLOT UPSAMPLED FFT
    upsampled_fft = 20 * np.log10(abs(np.fft.fft(pulses[0])))
    freqs = np.fft.fftfreq(pulses[0].shape[-1])
    w = (2 * np.pi) * np.fft.fftshift(freqs)
    axis[1].semilogy(w, upsampled_fft.real, ".:")
    axis[1].set_title("b_1 Upsampled by 20 Frequency Response")

    fig.supylabel("DFT Magnitude (dB)")
    fig.supxlabel("frequency (radians / sample)")
    plt.show()


def TEST_PlotDFTFindUpSamplingLPF():
    pulses = ApplyPulseShaping(TEST_HELPER_GetRandomBitSequenceTruncated())
    pulses, h = UpSample20WithLPF(pulses)

    response = np.fft.fft(h)
    freq = np.fft.fftshift(np.fft.fftfreq(response.shape[-1]))
    response = np.fft.fftshift(response)
    w = 2 * np.pi * freq
    plt.semilogy(w, abs(response))
    plt.ylabel("DFT Magnitude")
    plt.xlabel("frequency (radians / sample)")
    plt.title("Custom LPF Frequency Response")
    plt.show()


def TEST_PlotUpSampledPulses():
    pulses = ApplyPulseShaping(TEST_HELPER_GetRandomBitSequenceTruncated())
    pulses = UpSample20WithLPF(pulses)

    fig, axis = plt.subplots(2)

    axis[0].plot(np.arange(pulses[0].shape[-1]), pulses[0], ",-")
    axis[1].plot(np.arange(pulses[1].shape[-1]), pulses[1], ",-")
    axis[0].set_title("b_1[n] After Upsampling by 20")
    axis[1].set_title("b_2[n] After Upsampling by 20")

    fig.supxlabel("n")
    plt.show()


def TEST_PlotDFTMModulationSignal():
    pulses = ApplyPulseShaping(TEST_HELPER_GetRandomBitSequenceTruncated())
    pulses, _ = UpSample20WithLPF(pulses)

    response = np.fft.fft(pulses[0])
    freq = np.fft.fftshift(np.fft.fftfreq(response.shape[-1]))
    w = 2 * np.pi * freq
    response = np.fft.fftshift(response)
    plt.semilogy(w, abs(response))
    plt.ylabel("DFT Magnitude")
    plt.xlabel("frequency (radians / sample)")
    plt.title("")
    plt.show()


def TEST_PlotTransmitSignal():
    transmit_signal = Transmit(TEST_HELPER_GetRandomBitSequenceTruncated())

    carrier_frequency = 0.44 * np.pi  # radians/sample
    n = np.arange(np.floor(len(transmit_signal) / 10))
    carrier_signal = np.cos(carrier_frequency * n) + np.sin(carrier_frequency * n)

    plt.plot(
        n,
        transmit_signal[0 : n.shape[-1]],
        ",-",
        label="Transmitted Signal (With Modulation)",
    )
    plt.plot(n, carrier_signal, ",-", label="Carrier Signal (No Modulation)")
    plt.legend()

    plt.xlabel("n")
    plt.show()


if __name__ == "__main__":
    # TEST_PlotHDTFT()
    # TEST_PlotHDFT()
    # TEST_PlotImpulseResponseFunction()
    # TEST_PlotFIR()
    # TEST_ApplyPulseShaping()
    # TEST_ApplyPulseShapingWithUpSampling()
    TEST_PlotDFTFindUpSamplingLPF()
    # TEST_PlotUpSampledPulses()
    # TEST_PlotTransmitSignal()
    TEST_PlotDFTMModulationSignal()
