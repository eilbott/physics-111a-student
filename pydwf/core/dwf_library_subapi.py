'''The |pydwf.core.dwf_library_subapi| module provides a single class: |AbstractDwfLibrarySubApi|.'''

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydwf.core.dwf_library import DwfLibrary


class AbstractDwfLibrarySubAPI:
    '''Abstract base class for the sub-API attributes of a |DwfLibrary|.'''

    def __init__(self, dwf: DwfLibrary):
        self._dwf = dwf

    @property
    def dwf(self):
        '''Return the |DwfLibrary| instance of which we are an attribute.

        Returns:
            DwfLibrary: The |DwfLibrary| instance.
        '''
        return self._dwf

    @property
    def lib(self):
        '''Return the *ctypes* library.

        :meta private:

        Returns:
            ctypes.CDLL: The *ctypes* library.
        '''
        return self.dwf.lib
