'''The |pydwf.core.dwf_library| module provides a single class: |DwfLibrary|.'''

import sys
import ctypes

from pydwf.core.auxiliary.dwf_function_signatures import dwf_function_signatures, dwf_version as expected_dwf_version
from pydwf.core.auxiliary.exceptions import PyDwfError, DwfLibraryError
from pydwf.core.auxiliary.enum_types import DwfErrorCode, DwfDeviceParameter
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.api.device_enumeration import DeviceEnumeration
from pydwf.core.api.device_control import DeviceControl
from pydwf.core.api.spectrum import Spectrum


class DwfLibrary:
    '''The |DwfLibrary| class provides access to miscellaneous library functionality through the handful of methods
    it provides, and to |device enumeration:link|, |device control:link|, and |signal processing:link| functionality
    via its |deviceEnum:link|, |deviceControl:link|, and |spectrum:link| attributes.

    .. rubric:: |DwfLibrary| attributes

    Attributes:
        deviceEnum (DeviceEnumeration):
            Provides access to the |device enumeration:link| functionality.
        deviceControl (DeviceControl):
            Provides access to the |device control:link| functionality.
        spectrum (Spectrum):
            Provides access to the |signal processing:link| functionality.

    .. rubric:: |DwfLibrary| methods
    '''

    def __init__(self, check_library_version: bool = False) -> None:
        '''Initialize a |DwfLibrary| instance.

        A single |DwfLibrary| instance should be created by a user of |pydwf| to serve as an entry point to all
        |pydwf| functionality.

        When initializing a |DwfLibrary|, the shared library |libdwf| on top of which |pydwf| is built is loaded into
        memory using Python's standard |ctypes:link| module.

        A version check can be enabled to make sure that the shared library version corresponds exactly to the version
        that was used while developing and testing the current |pydwf| version (|libdwf-version|), and an exception is
        raised if a mismatch is detected. Enabling this flag is recommended for critical applications.

        After passing the version check (if enabled), the functions provided by the shared library are type-annotated.
        This means that calls into the shared library with incompatible parameter types will raise an exception.
        This mechanism helps to catch many bugs while using |pydwf|.

        As a last initialization step, the :py:attr:`deviceEnum`, :py:attr:`deviceControl`, and :py:attr:`spectrum`
        attributes are initialized. They can be used by a user program to access the |device enumeration:link|,
        |device control:link| functionality, and |signal processing:link| functionality.

        Parameters:
            check_library_version (bool): If True, the version number of the C library will be checked against the
                version of the C library from which the type information used by |pydwf| was derived. In case of a
                mismatch, an exception will be raised.

        Raises:
            PyDwfError: The version check could not be performed due to an unexpected low-level error while querying
                the shared library version, or a version mismatch was detected.
        '''

        try:
            if sys.platform.startswith('win'):
                lib = ctypes.cdll.dwf
            elif sys.platform.startswith('darwin'):
                lib = ctypes.cdll.LoadLibrary('/Library/Frameworks/dwf.framework/dwf')
                # SB: The following does not circumvent the library loading issue on Mac.
                #     The path '/Library/Frameworks/dwf.framework/dwf' seems to be expected
                #     internally as well as here explicitly.
                # try:
                #     lib = ctypes.cdll.LoadLibrary('/Library/Frameworks/dwf.framework/dwf')
                # except OSError:
                #     lib = ctypes.cdll.LoadLibrary('/Applications/WaveForms.app/Contents/Frameworks/dwf.framework/dwf')
            else:
                lib = ctypes.cdll.LoadLibrary('libdwf.so')
        except OSError as exception:
            # Modified for UC Berkeley PHYSICS 111A
            message = 'Unable to open DWF library using ctypes.'
            if sys.platform.startswith('darwin'):
                message += (
                    '\nIt looks like you are using a Mac. '
                    'Running the following in a terminal should fix this issue:\n'
                    '`sudo ln -s /Applications/WaveForms.app/Contents/Frameworks/dwf.framework '
                    '/Library/Frameworks/`.'
                    '\nYou will be prompted for your computer login password.'
                )
            raise PyDwfError(message) from exception

        if check_library_version:
            # Note that the 'FDwfGetVersion()' function has not yet been type-annotated here.
            c_version = ctypes.create_string_buffer(32)
            result = lib.FDwfGetVersion(c_version)
            if result != RESULT_SUCCESS:
                raise PyDwfError('Unable to verify library version.')
            actual_dwf_version = c_version.value.decode()
            if actual_dwf_version != expected_dwf_version:
                raise PyDwfError(
                    f'DWF library version mismatch: pydwf module expects {expected_dwf_version},'
                    f' but actual library is version {actual_dwf_version}.'
                )

        self._annotate_function_signatures(lib)

        self._lib = lib

        # Initialize sub-API instances and assign them to attributes.

        self.deviceEnum = DeviceEnumeration(self)
        self.deviceControl = DeviceControl(self)
        self.spectrum = Spectrum(self)

    @property
    def lib(self) -> ctypes.CDLL:
        '''Return the |ctypes:link| shared library instance used to access the DWF library.

        This property is provided for internal |pydwf| use.

        Returns:
            ctypes.CDLL: The |ctypes| library.

        :meta private:
        '''
        return self._lib

    @staticmethod
    def _annotate_function_signatures(lib: ctypes.CDLL) -> None:
        '''Add |ctypes:link| return type and parameter type annotations for all known functions in the DWF
        shared library.

        This method uses a list of DWF function signatures derived from the *dwf.h* C header file to perform
        the type annotation.

        Parameters:
            lib (ctypes.CDLL): The shared library whose functions will be type-annotated.

        This method is used exclusively by the :py:meth:`__init__` method.
        '''

        function_signatures = dwf_function_signatures(typespec_ctypes)

        for (name, restype, argtypes, UNUSED_obsolete_flag) in function_signatures:
            argtypes = [argtype for (argname, argtype) in argtypes]
            try:
                func = getattr(lib, name)
                func.restype = restype
                func.argtypes = argtypes
            except AttributeError:
                # Do not annotate functions that are not present in the shared library.
                # This can happen, for example, when pydwf is used with an older version of the shared library.
                pass

    def exception(self) -> DwfLibraryError:
        '''Return an exception describing the most recent error.

        This method is used by |pydwf| to generate a descriptive |DwfLibraryError:link| exception
        in case a DWF C library function has just failed.

        This method should not be called by |pydwf| users. It is for internal |pydwf| use only.

        Note that this method *returns* an exception instance; it doesn't *raise* it.

        Returns:
            DwfLibraryError: An exception describing the error reported by the last DWF library call.

        :meta private:
        '''
        return DwfLibraryError(self.getLastError(), self.getLastErrorMsg())

    def getLastError(self) -> DwfErrorCode:
        '''Retrieve the last error code in the calling process.

        The error code is cleared when other API functions are called and is only set when an API function
        fails during execution.

        Note:
            When using |pydwf| there is no need to call this method directly, since low-level errors reported by
            the C library are automatically converted to a |DwfLibraryError:link| exception, which includes
            both the error code and the corresponding message.

        Returns:
            DwfErrorCode: The DWF error code of last API call.

        Raises:
            DwfLibraryError: the last error code cannot be retrieved.
        '''
        c_dwferc = typespec_ctypes.DwfErrorCode()
        result = self._lib.FDwfGetLastError(c_dwferc)
        if result != RESULT_SUCCESS:
            # If the 'FDwfGetLastError()' call itself fails, we cannot get a proper error code or message.
            raise DwfLibraryError(None, 'FDwfGetLastError() failed.')
        dwferc = DwfErrorCode(c_dwferc.value)
        return dwferc

    def getLastErrorMsg(self) -> str:
        '''Retrieve the last error message.

        The error message is cleared when other API functions are called and is only set when an API function
        fails during execution.

        Note:
            When using |pydwf| there is no need to call this method directly, since low-level errors reported by
            the C library are automatically converted to a |DwfLibraryError:link| exception, which includes
            both the error code and the corresponding message.

        Returns:
            str: The error message of the last API call.

            The string may consist of multiple messages, separated by a newline character,
            that describe the events leading to the error.

        Raises:
            DwfLibraryError: The last error message cannot be retrieved.
        '''
        c_error_message = ctypes.create_string_buffer(512)
        result = self._lib.FDwfGetLastErrorMsg(c_error_message)
        if result != RESULT_SUCCESS:
            raise DwfLibraryError(None, 'FDwfGetLastErrorMsg() failed.')
        error_message = c_error_message.value.decode()
        return error_message

    def getVersion(self) -> str:
        '''Retrieve the library version string.

        Returns:
            str: The version of the DWF C library, composed of major, minor, and build numbers
            (e.g., '|libdwf-version|').

        Raises:
            DwfLibraryError: The library version string cannot be retrieved.
        '''
        c_version = ctypes.create_string_buffer(32)
        result = self._lib.FDwfGetVersion(c_version)
        if result != RESULT_SUCCESS:
            raise self.exception()
        version = c_version.value.decode()
        return version

    def paramSet(self, device_parameter: DwfDeviceParameter, value: int) -> None:
        '''Configure a default device parameter value.

        Device parameters are settings of a specific |DwfDevice|.
        Refer to the |device parameters:link| section for more information.

        This method sets a default device parameter value to be used for devices that are
        opened subsequently.

        See Also:
            To set the parameter value of a specific |DwfDevice|, use the
            :py:meth:`DwfDevice.paramSet <pydwf.core.DwfDevice.paramSet>` method.

        Warning:
            The device parameter values are not checked to make sure they correspond to a valid
            value for the specific device parameter.

        Parameters:
            device_parameter (DwfDeviceParameter): The device parameter for which to set the default value.
            value (int): The default device parameter value.

        Raises:
            DwfLibraryError: The device parameter value cannot be set.
        '''
        result = self._lib.FDwfParamSet(device_parameter.value, value)
        if result != RESULT_SUCCESS:
            raise self.exception()

    def paramGet(self, device_parameter: DwfDeviceParameter) -> int:
        '''Return a default device parameter value.

        Device parameters are settings of a specific |DwfDevice|.
        Refer to the |device parameters:link| section for more information.

        This method retrieves device parameter values at the library level (i.e., not tied to a specific device).
        They are used as default device parameter values for devices that are opened subsequently.

        See Also:
            To get the parameter value of a specific |DwfDevice|, use the
            :py:meth:`DwfDevice.paramGet <pydwf.core.DwfDevice.paramGet>` method.

        Parameters:
            device_parameter (DwfParameter): The device parameter for which to get the value.

        Returns:
            int: The retrieved device parameter value.

        Raises:
            DwfLibraryError: The device parameter value cannot be retrieved.
        '''
        c_value = typespec_ctypes.c_int()
        result = self._lib.FDwfParamGet(device_parameter.value, c_value)
        if result != RESULT_SUCCESS:
            raise self.exception()
        value = c_value.value
        return value
