import scipy
from scipy.special import sinc  # sinc(x) = sin(pi*x)/(pi*x)
import numpy as np
import matplotlib.pyplot as plt


def FIRWindowMethod(n_taps, w_p_low, w_p_high, w_transition_width):
    # Input:
    #   - numtaps (N-samples of FIR filter)
    #   - w_p_low (lower pass band frequency where magnitude is 0.5)
    #   - w_p_high (higher pass band frequency where magnitude is 0.5)
    #   - w_transition_width controls the width of the transition region; sharper
    #     (ie smaller) transitions increase ripple

    # Find the kaiser window shape:
    beta = scipy.signal.kaiser_beta(
        scipy.signal.kaiser_atten(n_taps, w_transition_width)
    )
    kaiser_window = scipy.signal.windows.get_window(
        ("kaiser", beta), n_taps, fftbins=False
    )

    # Plot kaiser window
    # i = np.arange(0, len(kaiser_window))
    # plt.figure()
    # plt.plot(i, kaiser_window)
    # plt.show()

    # Make specified frequencies (rad/sample) to (1/sample)
    w_p_low = w_p_low / (2*np.pi)
    w_p_high = w_p_high / (2*np.pi)

    # Construct IIR Ideal Filter (pre-truncate--ie rect() window--for efficiency)
    sinc_shift = 0.5 * (n_taps - 1)
    x = np.arange(0, n_taps) - sinc_shift
    fs_period = (x[0] - x[-1]) / len(x)
    fs = 1 / fs_period
    h = (fs * 2 * w_p_high * np.sinc(fs * 2 * w_p_high * x)) - (
        fs * 2 * w_p_low * np.sinc(fs * 2 * w_p_low * x)
    )

    # Apply window shaping to pre-truncated Ideal IIR (which is really an FIR at this point)
    h = h * kaiser_window

    return h
