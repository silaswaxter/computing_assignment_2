import numpy as np


def FindRandom4BitSymbols(symbol_count):
    """Generates a uniformly random symbol_countx2 array where each element is either 1 or -1.

    Args:
        symbol_count: the number of symbols to generate.

    Returns:
        A uniformly random symbol_countx2 array where each element is either 1 or -1
    """
    random_number_generator = np.random.default_rng()
    zero_ones = random_number_generator.integers(low=0, high=2, size=(symbol_count, 2))
    for i in range(len(zero_ones)):
        for j in range(2):
            if zero_ones[i][j] == 0:
                zero_ones[i][j] = -1
    output = zero_ones
    return output


# Quick and dirty visual test to check that distribution is expected
if __name__ == "__main__":
    sample_count = 1000000
    bit_count = sample_count * 2
    test_data = FindRandom4BitSymbols(sample_count)

    # Count the number of +1 and -1
    plus_one_count = 0
    negative_one_count = 0
    for i in range(len(test_data)):
        for j in range(2):
            if test_data[i][j] == 1:
                plus_one_count += 1
            if test_data[i][j] == -1:
                negative_one_count += 1

    print("Expected -1 occurance: 50%")
    print("Expected +1 occurance: 50%")
    print("Actual -1 occurance: %0.2f%%" % ((negative_one_count / bit_count) * 100))
    print("Actual +1 occurance: %0.2f%%" % ((plus_one_count / bit_count) * 100))
