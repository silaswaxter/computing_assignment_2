import numpy as np


def DownSample(input_signal, factor):
    """
    Down samples input_signal, returning the input signal with every factor multiple
    value

    Args:
        input_signal: the signal to downsample
        factor: an integer describing how much downsampling should be performed
    Returns:
        The upsampled input_signal.
    """
    output_shape = list(input_signal.shape)
    output_shape[0] = int(output_shape[0] / factor)
    output_shape = tuple(output_shape)
    output_signal = np.zeros(output_shape)
    for i in range(len(output_signal)):
        output_signal[i] = input_signal[i * factor]
    return output_signal


def UpSample(input_signal, factor):
    """
    Upsamples input_signal by factor by placing factor - 1 zeros in between values of
    the input_signal

    Args:
        input_signal: the signal to upsample
        factor: an integer describing how much upsampling should be performed
    Returns:
        The upsampled input_signal.
    """
    output_shape = list(input_signal.shape)
    output_shape[0] *= factor
    output_shape = tuple(output_shape)
    output_signal = np.zeros(output_shape)

    for i in range(len(input_signal)):
        output_signal[i * factor] = input_signal[i]

    return output_signal


def TEST_DownSample():
    test_x = np.array(
        [
            [-1, 1],
            [0xBAD, 0xBAD],
            [1, 1],
            [0xBAD, 0xBAD],
            [1, -1],
            [0xBAD, 0xBAD],
            [1, -1],
            [0xBAD, 0xBAD],
            [1, -1],
            [0xBAD, 0xBAD],
        ]
    )
    test_x_up = DownSample(test_x, 2)
    print("Test Input:")
    print(test_x)
    print("Downsampled-by-2 Output:")
    print(test_x_up)


def TEST_UpSample():
    test_x = np.array(
        [
            [-1, 1],
            [1, -1],
            [1, 1],
            [1, -1],
            [1, -1],
            [1, 1],
            [1, -1],
            [1, 1],
            [1, -1],
            [1, 1],
        ]
    )
    test_x_up = UpSample(test_x, 4)
    print("Test Input:")
    print(test_x)
    print("Upsampled-by-4 Output:")
    print(test_x_up)


if __name__ == "__main__":
    TEST_UpSample()
    TEST_DownSample()
