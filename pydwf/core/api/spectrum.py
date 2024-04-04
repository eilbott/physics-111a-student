'''The |pydwf.core.api.spectrum| module provides a single class: |Spectrum|.'''

from typing import Optional, Tuple

import numpy as np

from pydwf.core.dwf_library_subapi import AbstractDwfLibrarySubAPI

from pydwf.core.auxiliary.enum_types import DwfWindow

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS


class Spectrum(AbstractDwfLibrarySubAPI):
    '''The |Spectrum| class provides access to the signal processing functionality of a |DwfLibrary:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfLibrary| and subsequently assigned to its
        public |spectrum:link| attribute for access by the user.
    '''

    def window(self, n: int, winfunc: DwfWindow, beta: Optional[float] = None) -> Tuple[np.ndarray, float]:
        '''Return the window coefficients for the specified window, and the window's noise-equivalent bandwidth.

        Parameters:
            n (int): The length of the window to be generated.
            winfunc (DwfWindow): The type of window to be generated.
            beta (Optional[float]): Parameter for the Kaiser window; unused for other windows. If not specified,
               a value of 0.5 is used.

        Returns:
           Tuple[np.ndarray, float]: A two-element tuple. The first element of the tuple is a numpy array of length
           `n`, containing the window coefficients. All windows supported are symmetric, and they are scaled to have
           the sum of their values equal to `n`.

           The second element of the tuple is the noise equivalent bandwidth of the signal. For a coefficient array
           `w`, this value is equal to `len(w) * np.sum(w**2) / np.sum(w)**2`.

        Note:
           When working in Python, it may be better to use the functionality provided in the scipy.signal.windows
           package instead.
        '''

        if beta is None:
            beta = 0.5

        c_nebw = typespec_ctypes.c_double()
        w = np.empty(n, dtype=np.float64)
        result = self.lib.FDwfSpectrumWindow(
            w.ctypes.data_as(typespec_ctypes.c_double_ptr), n, winfunc.value, beta, c_nebw
        )
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        nebw = c_nebw.value
        return (w, nebw)

    def fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Perform a Fast Fourier Transform.

        Parameters:
            data: The real-valued data to be transformed. This must be a 1-dimensional array with a length that is a
               power of 2.

        Returns:
           Tuple[np.ndarray, np.ndarray]: A tuple (magnitude, phase); both the magnitude and phase arrays are
           real-valued 1D numpy arrays with the same length as the input array `data`. The first array contains
           the non-negative magnitude of the signal for this particular frequency; the second array contains its phase.

        Note:
            The scaling of the FFT calculated is unusual. Compared to Matlab, numpy, and most other implementations, it
            is scaled by a factor (2 / n), with n the number of points in the input array.

            To get from the (magnitude, phase) representation as returned by this function to a vector of complex
            number that is comparable to what e.g. the `numpy.fft.rfft` function returns, you can do:

            z = (len(magnitude) / 2.0) * magnitude * np.exp(phase * 1j)

            When working in Python, it may be better to simply use the numpy.fft.rfft() function directly.
            It is faster and provides support for data vectors with a length that is not a power of two.
        '''

        n = len(data)

        num_bins = n // 2 + 1

        bins = np.empty(num_bins, dtype=np.float64)
        phase = np.empty(num_bins, dtype=np.float64)

        result = self.lib.FDwfSpectrumFFT(
            data.ctypes.data_as(typespec_ctypes.c_double_ptr),
            n,
            bins.ctypes.data_as(typespec_ctypes.c_double_ptr),
            phase.ctypes.data_as(typespec_ctypes.c_double_ptr),
            num_bins
        )

        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        return(bins, phase)

    def transform(self, data: np.ndarray, num_bins: int,
                  first_freq: float, last_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        '''Perform a Chirp-Z transform.

        Parameters:
            data: The real-valued data to be transformed.

            num_bins (int): The number of bins to be calculated.
            
            first_freq (float): The frequency of the first bin, scaled to 0.5 * sample_frequency.
            Should be between 0 and 1, inclusive.

            last_freq (float): The frequency of the last bin, scaled to 0.5 * sample_frequency.
            Should be between 0 and 1, inclusive.

        Returns:
           Tuple[np.ndarray, np.ndarray]: A tuple (magnitude, phase); both the magnitude and phase arrays are
           real-valued 1D numpy arrays with the same length as the input array `data`. The first array contains
           the non-negative magnitude of the signal for this particular frequency; the second array contains its phase.

        Note:
            The scaling of the FFT calculated is unusual. Compared to Matlab, numpy, and most other implementations, it
            is scaled by a factor (2 / n), with n the number of points in the input array.

            To get from the (magnitude, phase) representation as returned by this function to a vector of complex
            number that is comparable to what e.g. the `scipy.signal.czt` function returns, you can do:

            z = (len(data) / 2.0) * magnitude * np.exp(phase * 1j)

            When working in Python, it may be better to simply use the scipy.signal.czt() function directly.

        '''

        n = len(data)

        bins = np.empty(num_bins, dtype=np.float64)
        phase = np.empty(num_bins, dtype=np.float64)

        result = self.lib.FDwfSpectrumTransform(
                data.ctypes.data_as(typespec_ctypes.c_double_ptr),
                n,
                bins.ctypes.data_as(typespec_ctypes.c_double_ptr),
                phase.ctypes.data_as(typespec_ctypes.c_double_ptr),
                num_bins,
                first_freq,
                last_freq
            )
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        return (bins, phase)
