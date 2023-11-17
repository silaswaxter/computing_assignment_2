import fir_window_designer
import numpy as np
import matplotlib.pyplot as plt
import convolution
import simulated_analog_channel
import transmitter
import upsampler


def GetInputSignalBandwidth():
    w_p_center = 0.44 * np.pi
    w_p_cutoff = (
        3 * np.pi / 8
    ) / 20  # original cutoff with transmitter-20-upsample adjustment
    return (w_p_center - w_p_cutoff), (w_p_center + w_p_cutoff)


def GetInputBandPassFilter():
    w_p_low, w_p_high = GetInputSignalBandwidth()
    return fir_window_designer.FIRWindowMethod(255, w_p_low, w_p_high, 0.1)


def ApplyInputBandPassFilter(input):
    return convolution.Convolve_FiniteSequences(input, GetInputBandPassFilter())


def SimulatedADCWith8XUpsample(input):
    upsample_factor = 10
    # TODO: downsample instead of upsample
    upsamped_input = upsampler.UpSample(input, upsample_factor)
    # remove aliasing by with filter
    w_p_low, w_p_high = GetInputSignalBandwidth()
    w_p_low /= upsample_factor
    w_p_high /= upsample_factor
    h_lpf = fir_window_designer.FIRWindowMethod(255, 0, w_p_high, 0.1)
    pass_band_gain = upsample_factor
    h_lpf *= pass_band_gain
    response = np.fft.fft(h_lpf)
    response = np.fft.fftshift(response)
    freq = np.fft.fftshift(np.fft.fftfreq(h_lpf.shape[-1]))
    w = 2 * np.pi * freq
    # plt.figure()
    # plt.plot(w, abs(response))
    # plt.title("mag freq resp")
    # plt.show()
    return convolution.Convolve_FiniteSequences(upsamped_input, h_lpf)


def Demodulate(input):
    demodulated = np.zeros((2, len(input)))
    w_1 = ((3 * np.pi / 8) / 20) / 8
    demodulated[0] = np.cos(w_1 * np.arange(0, len(input))) * input
    demodulated[1] = np.sin(w_1 * np.arange(0, len(input))) * input
    response_0 = np.fft.fft(demodulated[0])
    response_0 = np.fft.fftshift(response_0)
    response_1 = np.fft.fft(demodulated[1])
    response_1 = np.fft.fftshift(response_1)
    freq = np.fft.fftshift(np.fft.fftfreq(response_0.shape[-1]))
    w = 2 * np.pi * freq
    plt.figure()
    plt.plot(w, abs(response_0))
    plt.plot(w, abs(response_1))
    plt.title("")
    plt.show()


def TEST_HELPER_receiver_input_generator():
    test_channel = simulated_analog_channel.SimulatedAnalogChannel(
        gaussian_noise_constant_gain=0.1
    )
    test_data = np.array(
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
        ]
    )
    test_transmit_signal = transmitter.Transmit(test_data)
    return test_channel.ReceivedSignal(test_transmit_signal)


def TEST_Demodulate():
    test_input_filtered_signal = ApplyInputBandPassFilter(
        TEST_HELPER_receiver_input_generator()
    )
    upsampled = SimulatedADCWith8XUpsample(test_input_filtered_signal)
    Demodulate(upsampled)


def TEST_SimulatedADCWith8XUpsample():
    test_input_filtered_signal = ApplyInputBandPassFilter(
        TEST_HELPER_receiver_input_generator()
    )
    upsampled = SimulatedADCWith8XUpsample(test_input_filtered_signal)
    response = np.fft.fft(upsampled)
    response = np.fft.fftshift(response)
    freq = np.fft.fftshift(np.fft.fftfreq(upsampled.shape[-1]))
    w = 2 * np.pi * freq

    plt.figure()
    plt.plot(np.arange(0, len(upsampled)), upsampled)
    plt.title("h[n]")

    plt.figure()
    plt.plot(w, abs(response))
    plt.title("mag freq resp")
    plt.show()


def TEST_PlotInputBandPassFilter():
    h = GetInputBandPassFilter()
    response = np.fft.fft(h)
    response = np.fft.fftshift(response)
    freq = np.fft.fftshift(np.fft.fftfreq(h.shape[-1]))
    w = 2 * np.pi * freq

    plt.figure()
    plt.plot(np.arange(0, len(h)), h)
    plt.title("h[n]")

    plt.figure()
    plt.plot(w, abs(response))
    plt.title("mag freq resp")
    plt.show()


def TEST_PlotBandpassFilteredTestData():
    test_input_filtered_signal = ApplyInputBandPassFilter(
        TEST_HELPER_receiver_input_generator()
    )
    response = np.fft.fft(test_input_filtered_signal)
    response = np.fft.fftshift(response)
    freq = np.fft.fftshift(np.fft.fftfreq(test_input_filtered_signal.shape[-1]))
    w = 2 * np.pi * freq

    plt.figure()
    plt.plot(np.arange(0, len(test_input_filtered_signal)), test_input_filtered_signal)
    plt.title("x[n]")

    plt.figure()
    plt.plot(w, abs(response))
    plt.title("mag freq resp")
    plt.show()


if __name__ == "__main__":
    TEST_PlotInputBandPassFilter()
    TEST_PlotBandpassFilteredTestData()
    TEST_SimulatedADCWith8XUpsample()
    TEST_Demodulate()
