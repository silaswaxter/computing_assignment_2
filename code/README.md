# How to run code

The code is written for Python 3.9.16 and uses a couple other packages:

- numpy
  - makes working with math easier
  - contains FFT and IFFT algorithms
- matplotlib
  - generate plots from sequences
- scipyversion
  - contains algorithms for designing LPF

This version of python is the version installed on OSU's engineering servers.
`numpy` and `matplotlib` versions were selected so that they match the
engineering servers; `scipy` was not. To ensure the exact same versions of
packages are installed. Run `pip install -r requirements.txt` where
`requirements.txt` is the file in this directory.

## About the code

At the bottom of each of my "modules" is a simple set of tests that are run when
the file is executed. You can enable and disable tests by
commenting/uncommenting the function calls.
