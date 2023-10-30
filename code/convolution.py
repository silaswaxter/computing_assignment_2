import numpy as np


def Convolve_FiniteSequences(x, h):
    """An example implementation of an FIR filter

    Args:
        x: The input signal to manipulate with the FIR system.
        h: The impulse response function

    Returns:
        The output signal from the FIR system
    """

    # see https://dsp.stackexchange.com/a/34922
    output_length = len(x) + len(h) - 1
    output = np.zeros(output_length)

    # During this convolution implementation, h is flipped; so, the leftest
    # possible index will be whatever negative of whatever h's largest index is.
    h_left_most_index = -(len(h) - 1)
    x_right_most_index = len(x) - 1
    for n in range(0, len(output)):
        left_overlap_index = 0
        h_shifted_left_most_index = h_left_most_index + n
        if h_shifted_left_most_index > 0:
            left_overlap_index = h_shifted_left_most_index

        right_overlap_index = n
        if right_overlap_index > x_right_most_index:
            right_overlap_index = x_right_most_index

        overlap_length = right_overlap_index - left_overlap_index + 1

        for j in range(0, overlap_length):
            m = left_overlap_index + j
            output[n] += x[m] * h[n - m]

    return output


# Quick and Dirty Test Methods:
def TEST_ArbitraryLenXEqualLenH():
    x = np.array([1, 2, 3])
    h = np.array([1, 2, 3])

    expected_output = np.array([1, 4, 10, 12, 9])
    actual_output = Convolve_FiniteSequences(x, h)

    for i in range(0, len(expected_output)):
        if expected_output[i] != actual_output[i]:
            print(
                "expected_output[{0}]={1} != actual_output[{0}]={2}".format(
                    i, expected_output[i], actual_output[i]
                )
            )

            return False

    return True


def TEST_ArbitraryLenXGreaterThanLenH():
    x = np.array([2, 2, 4, 4])
    h = np.array([1, 1])

    expected_output = np.array([2, 4, 6, 8, 4])
    actual_output = Convolve_FiniteSequences(x, h)

    for i in range(0, len(expected_output)):
        if expected_output[i] != actual_output[i]:
            print(
                "expected_output[{0}]={1} != actual_output[{0}]={2}".format(
                    i, expected_output[i], actual_output[i]
                )
            )

            return False

    return True


def TEST_ArbitraryLenXLessThanLenH():
    x = np.array([2, 2])
    h = np.array([1, 1, 1, 1])

    expected_output = np.array([2, 4, 4, 4, 2])
    actual_output = Convolve_FiniteSequences(x, h)

    for i in range(0, len(expected_output)):
        if expected_output[i] != actual_output[i]:
            print(
                "expected_output[{0}]={1} != actual_output[{0}]={2}".format(
                    i, expected_output[i], actual_output[i]
                )
            )

            return False

    return True


def TEST_XAsImpulseResponseReturnsH():
    x = np.array([1])
    h = np.array([-1, 0, 5])

    expected_output = np.array([-1, 0, 5])
    actual_output = Convolve_FiniteSequences(x, h)

    for i in range(0, len(expected_output)):
        if expected_output[i] != actual_output[i]:
            print(
                "expected_output[{0}]={1} != actual_output[{0}]={2}".format(
                    i, expected_output[i], actual_output[i]
                )
            )

            return False

    return True


if __name__ == "__main__":
    isAllTestsPassing = True
    isAllTestsPassing = TEST_ArbitraryLenXEqualLenH()
    isAllTestsPassing = TEST_ArbitraryLenXGreaterThanLenH()
    isAllTestsPassing = TEST_ArbitraryLenXLessThanLenH()
    isAllTestsPassing = TEST_XAsImpulseResponseReturnsH()
    if isAllTestsPassing:
        print("All Tests Passed.")
    else:
        print("One or More Tests Failed.")
