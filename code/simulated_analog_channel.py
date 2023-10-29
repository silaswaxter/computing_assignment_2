import numpy as np


class SimulatedAnalogChannel:
    def __init__(self, gaussian_noise_constant_gain=1.0, gaussian_noise_stddev=1.0):
        self.gaussian_noise_constant_gain = gaussian_noise_constant_gain
        self.gaussian_noise_stddev = gaussian_noise_stddev

    def GetGaussianNoise(self, n_samples=1):
        """Returns a list-like object of length n_samples where elements are
           randomly distributed according to the gaussian distribution
           manipulated by gaussian_noise_stddev and gaussian_noise_constant_gain.

        Args:
            n_samples: the number of samples to return

        Returns:
            see method description
        """
        random_number_generator = np.random.default_rng()
        return self.gaussian_noise_constant_gain * random_number_generator.normal(
            loc=0, scale=self.gaussian_noise_stddev, size=n_samples
        )

    def ReceivedSignal(self, transmitted_signal, noise_signal=None):
        """Returns the received signal of the transmitted_signal through this
           simulated analog channel.

        Args:
            transmitted_signal: the signal to receive through this channel

        Returns:
            the received signal
        """
        received_signal = np.zeros(len(transmitted_signal))
        if noise_signal is None:
            noise_signal = self.GetGaussianNoise(len(transmitted_signal))
        for i in range(len(transmitted_signal)):
            received_signal[i] = transmitted_signal[i] + noise_signal[i]
        return received_signal


# Quick and dirty test
import matplotlib.pyplot as plt

if __name__ == "__main__":
    test_channel = SimulatedAnalogChannel(gaussian_noise_constant_gain=0.25)

    x = np.linspace(0, 4 * np.pi, 100)
    test_trans_sig = np.sin(x)

    test_noise_sig = test_channel.GetGaussianNoise(len(test_trans_sig))

    test_rec_sig = test_channel.ReceivedSignal(test_trans_sig, test_noise_sig)

    plt.plot(x, test_trans_sig, "s-")
    plt.plot(x, test_noise_sig, "o:", x, test_rec_sig, "o:")
    plt.show()
