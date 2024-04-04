'''The |pydwf.core.dwf_device_subapi| module provides a single class: |AbstractDwfDeviceSubApi|.'''

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydwf.core.dwf_device import DwfDevice


class AbstractDwfDeviceSubAPI:
    '''Abstract base class for the sub-API attributes of a |DwfDevice|.'''

    def __init__(self, device: DwfDevice):
        self._device = device

    @property
    def device(self):
        '''Return the |DwfDevice| instance of which we are an attribute.

        This is useful if we have a variable that contains a reference to a |DwfDevice| attribute,
        but we need the |DwfDevice| itself.

        Returns:
            DwfDevice: The |DwfDevice| instance that this attribute belongs to.
        '''
        return self._device

    @property
    def hdwf(self):
        '''Return the HDWF device handle.

        :meta private:

        Returns:
            int: The HDWF device handle.
        '''
        return self.device.hdwf

    @property
    def dwf(self):
        '''Return the |DwfLibrary| instance.

        :meta private:

        Returns:
            DwfLibrary: The |DwfLibrary| instance.

        '''
        return self.device.dwf

    @property
    def lib(self):
        '''Return the *ctypes* library.

        :meta private:

        Returns:
            ctypes.CDLL: The *ctypes* library.
        '''
        return self.dwf.lib
