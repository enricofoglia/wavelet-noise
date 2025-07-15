Theoretical background
----------------------

The wavelet transform is a mathematical tool that was born out of the necessity to analyze signals in the time and frequency domain simultaneously. Contrary to the spectrogram, a.k.a. Gabor transform, the wavelet transform adapts the length of the analysis window to the frequency of the signal, in a way that can be considered optimal. 

The wavelet tranform analyzes the signal by convolving it with a set of wavelet functions, which are defined as scaled and translated versions of a mother wavelet :math:`\psi`:

.. math::

    \psi_{a,b}(t) = \frac{1}{\sqrt{a}} \psi\left(\frac{t-b}{a}\right)

where :math:`a` is the scale (or dilation) parameter and :math:`b` is the translation parameter. The wavelet transform of a signal :math:`f(t)` is then defined as:

.. math::

    \mathcal{W}\{f\}(a,b) = \int_{-\infty}^{\infty} f(t) \psi_{a,b}(t) \mathrm{d}t

Using both :math:`a` and :math:`b` as continuous parameters, one gets the so called continuous wavelet transform (CWT). In practice, the convolution is computed using the Fast Fourier Transform (FFT) algorithm, which allows for efficient computation of the wavelet coefficients with a cost of :math:`\mathcal{O}(N \log N)` where :math:`N` is the number of samples in the signal. The CWT is especially useful when analyzing a signal using the so called scaleogram, which is a 2D representation of the magnitude of the wavelet coefficients as a function of scale and time. However, it provides a redundant representation of the signal, as the :math:`\psi_{a,b}` functions are not orthogonal. 

To obtaiin a set of wavelets that form an orthogonal basis for the space of square integrable signals, one can discretize the scale and translation parameters in octaves. This leads to a set of wavelet functions defined as:

.. math::

    \psi_{j,k}(t) = \frac{1}{\sqrt{2^j}} \psi\left(\frac{t - k 2^j}{2^j}\right)

where :math:`j\in\mathbb{N}` is the octave index and :math:`k\in\mathbb{Z}` is the translation index. The signal can then be decomposed into a sum of wavelet coefficients as follows:

.. math::

    f(t) = \sum_{j\in\mathbb{N}}\sum_{k\in\mathbb{Z}} w_{j,k} \psi_{j,k}(t)

where the coefficients :math:`w_{j,k}` can be computed via an inner product as:

.. math::

    w_{j,k} = \int_{-\infty}^{\infty} f(t) \psi_{j,k}(t) \mathrm{d}t = \langle f, \psi_{j,k} \rangle

The discrete wavelet transform (DWT) is the collection of the coefficients :math:`w_{j,k}`. 

.. important::
    Not all wavelet functions are othogonal. Finding wavelets that are at the same time orthogonal and well localized in the time and frequency domain is a non-trivial task, and has been a subject of intense research. The most popular wavelets today are the Daubechies wavelets and Morlet wavelets. In many examples, the Haar of the Shannon wavelets are used because of their simplicity. However, the Haar wavelet is not well localized in the frequency domain, and the opposite is true for the Shannon wavelet: both should be avoided in practice.

The DWT works by dividing the space of signals into subsets of lower and lower resolutions, creating a hierarchical nested structure known as multiresolution. This allows the DWT to be computed extremely efficiently by iteratively applying a low and a high pass filter to the signal, together with a downsampling operation. This process is known as the Mallat algorithm, of Fast Wavelet Transform (FWT), and it allows to compute the DWT in :math:`\mathcal{O}(N)` time.