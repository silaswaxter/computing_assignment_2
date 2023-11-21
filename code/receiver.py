import fir_window_designer
import numpy as np
import matplotlib.pyplot as plt
import convolution
import simulated_analog_channel
import transmitter
import factor_sampler
import test_helper


def GetInputSignalBandwidth():
    w_p_center = 0.44
    w_p_cutoff = (
        3 * np.pi / 8
    ) / 20  # original cutoff with transmitter-20-upsample adjustment
    return (w_p_center - w_p_cutoff), (w_p_center + w_p_cutoff)


def GetInputBandPassFilter():
    w_p_low, w_p_high = GetInputSignalBandwidth()
    return fir_window_designer.FIRWindowMethod(127, w_p_low, w_p_high, 0.1)


def ApplyInputBandPassFilter(input):
    return convolution.Convolve_FiniteSequences(input, GetInputBandPassFilter())


def GetSimulatedADCDownSampleRate():
    return 10


def SimulatedADC_Downsample(input):
    # downsample such that frequency is 8x the symbol rate. The symbol rate starts out
    # as 1 sample/second, then we upsample in the transmitter by a factor of 4 for pulse
    # shaping, then we upsample by a factor of 20 for simulated analog transmission. So
    # symbol rate before this ADC =1/(4*20)=1/80. Therefore, to get rate = 1/8, we
    # downsample (multiplication) by 10: 10*1/80=1/8.
    downsample_factor = GetSimulatedADCDownSampleRate()
    return factor_sampler.DownSample(input, downsample_factor)


def GetDemodulationFrequency():
    return (3 * np.pi / 8) / 2


def Demodulate(input):
    demodulated = np.zeros((2, len(input)))
    w_1 = GetDemodulationFrequency()
    demodulated[0] = np.cos(w_1 * np.arange(0, len(input))) * input
    demodulated[1] = np.sin(w_1 * np.arange(0, len(input))) * input
    return demodulated


def GetDemodulatedLPF():
    w_in_low, w_in_high = GetInputSignalBandwidth()
    # account for aliasing
    downsample_factor = GetSimulatedADCDownSampleRate()
    w_down_low = np.pi - (w_in_low * downsample_factor - np.pi)
    w_down_high = np.pi - (w_in_high * downsample_factor - np.pi)
    w_demodulate_high = max(np.pi / 2, w_down_low, w_down_high)

    # account for demodulation: sin(ax)f(x) or cos(ax)f(x) <=> (F(w-a) +- F(w+a))/2
    # the lpf cutoff frequency is the upper frequency of the left "image".
    w_demodulation = GetDemodulationFrequency()
    w_demodulate_high -= w_demodulation

    return fir_window_designer.FIRWindowMethod(127, 0, w_demodulate_high, 0.01)


def ApplyDemodulatedLPF(input):
    output = []
    for signal in input:
        output.append(convolution.Convolve_FiniteSequences(signal, GetDemodulatedLPF()))
    return output


def SubSampleToOriginalSymbolRate(input):
    output = []
    for signal in input:
        output.append(factor_sampler.DownSample(signal, 2))
    return output


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
    test_transmit_signal = transmitter.Transmit(test_data)
    return test_channel.ReceivedSignal(test_transmit_signal)


def Test_DemodulatedLPF():
    reciever_input = TEST_HELPER_receiver_input_generator()
    input_bpf_output = ApplyInputBandPassFilter(reciever_input)
    adc_output = SimulatedADC_Downsample(input_bpf_output)
    demodulated_output = Demodulate(adc_output)
    lpf_ouput = ApplyDemodulatedLPF(demodulated_output)

    lpf_output_dtf_0 = test_helper.GetDFTWithShiftsAndFrequencyScaling(lpf_ouput[0])
    lpf_output_dtf_1 = test_helper.GetDFTWithShiftsAndFrequencyScaling(lpf_ouput[1])

    _, axis = plt.subplots(2)
    axis[0].plot(lpf_output_dtf_0[0], abs(lpf_output_dtf_0[1]))
    axis[0].set(xlabel="Frequency (rad/sample)", ylabel="Magnitude")
    axis[0].set_title("LPF-Demodulated-Cosine Frequency Response")
    axis[1].plot(lpf_output_dtf_1[0], abs(lpf_output_dtf_1[1]))
    axis[1].set(xlabel="Frequency (rad/sample)", ylabel="Magnitude")
    axis[1].set_title("LPF-Demodulated-Sine Frequency Response")
    plt.show()


def Test_SubSampleToOriginalSymbolRate():
    reciever_input = TEST_HELPER_receiver_input_generator()
    input_bpf_output = ApplyInputBandPassFilter(reciever_input)
    adc_output = SimulatedADC_Downsample(input_bpf_output)
    demodulated_output = Demodulate(adc_output)
    lpf_ouput = ApplyDemodulatedLPF(demodulated_output)
    orignal_rate_downsampler_output = SubSampleToOriginalSymbolRate(lpf_ouput)
    plt.figure()
    plt.plot(
        np.arange(len(orignal_rate_downsampler_output[0])),
        orignal_rate_downsampler_output[0],
    )
    plt.title("Cosine-Demodulated Signal with Adjusted Rate vs. n")
    plt.show()

    orignal_rate_downsampler_output_dft_0 = (
        test_helper.GetDFTWithShiftsAndFrequencyScaling(
            orignal_rate_downsampler_output[0]
        )
    )
    orignal_rate_downsampler_output_dft_1 = (
        test_helper.GetDFTWithShiftsAndFrequencyScaling(
            orignal_rate_downsampler_output[1]
        )
    )

    _, axis = plt.subplots(2)
    axis[0].plot(
        orignal_rate_downsampler_output_dft_0[0],
        np.fft.fftshift(abs(orignal_rate_downsampler_output_dft_0[1])),
    )
    axis[1].plot(
        orignal_rate_downsampler_output_dft_0[0],
        np.fft.fftshift(abs(orignal_rate_downsampler_output_dft_1[1])),
    )
    axis[0].set_title("Cosine-Demodulated Signal with Adjusted Rate Frequency Response")
    axis[0].set(xlabel="Frequency (rad/sample)", ylabel="Magnitude")
    axis[1].set_title("Sine-Demodulated Signal with Adjusted Rate Frequency Response")
    axis[1].set(xlabel="Frequency (rad/sample)", ylabel="Magnitude")
    plt.show()


def TEST_Demodulate():
    reciever_input = TEST_HELPER_receiver_input_generator()
    input_bpf_signal = ApplyInputBandPassFilter(reciever_input)
    input_bpf_signal_dft = test_helper.GetDFTWithShiftsAndFrequencyScaling(
        input_bpf_signal
    )
    adc_output = SimulatedADC_Downsample(input_bpf_signal)
    demodulated_signal = Demodulate(adc_output)
    demodulated_signal_dtf_0 = test_helper.GetDFTWithShiftsAndFrequencyScaling(
        demodulated_signal[0]
    )
    demodulated_signal_dtf_1 = test_helper.GetDFTWithShiftsAndFrequencyScaling(
        demodulated_signal[1]
    )

    _, axis = plt.subplots(2)
    axis[0].plot(demodulated_signal_dtf_0[0], abs(demodulated_signal_dtf_0[1]))
    axis[0].set_title("Demodulated-Cosine Frequency Response")
    axis[0].set(xlabel="Frequency (rad/sample)", ylabel="Magnitude")
    axis[1].plot(demodulated_signal_dtf_1[0], abs(demodulated_signal_dtf_1[1]))
    axis[1].set(xlabel="Frequency (rad/sample)", ylabel="Magnitude")
    axis[1].set_title("Demodulated-Sine Frequency Response")
    plt.show()


def TEST_SimulatedADC_Downsampling():
    reciever_input = TEST_HELPER_receiver_input_generator()
    input_bpf_signal = ApplyInputBandPassFilter(reciever_input)
    input_bpf_signal_dft = test_helper.GetDFTWithShiftsAndFrequencyScaling(
        input_bpf_signal
    )

    output = SimulatedADC_Downsample(input_bpf_signal)
    output_dft = test_helper.GetDFTWithShiftsAndFrequencyScaling(output)

    _, axis = plt.subplots(2)
    axis[0].plot(np.arange(len(input_bpf_signal)), input_bpf_signal)
    axis[0].plot(np.arange(len(output)), output)
    axis[0].set_title("Downampler Input and Output Signals vs. n")
    axis[1].plot(input_bpf_signal_dft[0], abs(input_bpf_signal_dft[1]))
    axis[1].plot(output_dft[0], abs(output_dft[1]))
    axis[1].set_title("Downsampler Input and Output Signals Frequency Response vs. w")
    plt.show()


def TEST_PlotInputBandpassFilter():
    reciever_input = TEST_HELPER_receiver_input_generator()
    h_bpf = GetInputBandPassFilter()
    bpf_input = ApplyInputBandPassFilter(reciever_input)

    reciever_input_dft = test_helper.GetDFTWithShiftsAndFrequencyScaling(reciever_input)
    bpf_dft = test_helper.GetDFTWithShiftsAndFrequencyScaling(h_bpf)
    reciever_bpf_input_dft = test_helper.GetDFTWithShiftsAndFrequencyScaling(bpf_input)

    plt.figure()
    plt.plot(
        reciever_input_dft[0], abs(reciever_input_dft[1]), ":", label="Input Signal"
    )
    plt.plot(
        reciever_bpf_input_dft[0],
        abs(reciever_bpf_input_dft[1]),
        label="Band Pass Filtered Signal",
    )
    plt.plot(bpf_dft[0], 200 * abs(bpf_dft[1]), label="200x Band Pass Filter")
    plt.title("Frequency Response")
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # TEST_PlotInputBandpassFilter()
    # TEST_SimulatedADC_Downsampling()
    # TEST_Demodulate()
    # Test_DemodulatedLPF()
    Test_SubSampleToOriginalSymbolRate()
