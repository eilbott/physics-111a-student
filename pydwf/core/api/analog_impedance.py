'''The |pydwf.core.api.analog_impedance| module provides a single class: |AnalogImpedance|.'''

from typing import Tuple

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.enum_types import DwfState, DwfAnalogImpedance
from pydwf.core.auxiliary.constants import RESULT_SUCCESS


class AnalogImpedance(AbstractDwfDeviceSubAPI):
    '''The |AnalogImpedance| class provides access to the analog impedance measurement functionality of a
    |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently assigned to its public
        |analogImpedance:link| attribute for access by the user.
    '''

    # pylint: disable=too-many-public-methods

    def reset(self) -> None:
        '''Reset the |AnalogImpedance| functionality.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfAnalogImpedanceReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def configure(self, start: bool) -> None:
        '''Configure the |AnalogImpedance| functionality, and optionally start a measurement.

        Parameters:
            start (bool): Whether to start the measurement.

        Raises:
            DwfLibraryError: An error occurred while executing the *configure* operation.
        '''
        result = self.lib.FDwfAnalogImpedanceConfigure(self.hdwf, int(start))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def status(self) -> DwfState:
        '''Return the status of the |AnalogImpedance| functionality.

        Returns:
            DwfState: The status of the measurement.

        Raises:
            DwfLibraryError: An error occurred while executing the *status* operation.
        '''
        c_status = typespec_ctypes.DwfState()
        result = self.lib.FDwfAnalogImpedanceStatus(self.hdwf, c_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        status_ = DwfState(c_status.value)
        return status_

    def modeSet(self, mode: int) -> None:
        '''Set |AnalogImpedance| measurement mode.

        Parameters:
            mode (int): The measurement mode.

                The following modes are defined:

                * 0 — W1-C1-DUT-C2-R-GND
                * 1 — W1-C1-R-C2-DUT-GND
                * 8 — |Impedance analyzer:link| for the Analog Discovery

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceModeSet(self.hdwf, mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def modeGet(self) -> int:
        '''Get |AnalogImpedance| measurement mode.

        Returns:
            int: The measurement mode.

            The following modes are defined:

            * 0 — W1-C1-DUT-C2-R-GND
            * 1 — W1-C1-R-C2-DUT-GND
            * 8 — |Impedance analyzer:link| for the Analog Discovery

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_mode = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogImpedanceModeGet(self.hdwf, c_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        mode = c_mode.value
        return mode

    def referenceSet(self, reference: float) -> None:
        '''Set |AnalogImpedance| reference load value, in Ohms.

        Parameters:
            reference (float): The reference load, in Ohms.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceReferenceSet(self.hdwf, reference)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def referenceGet(self) -> float:
        '''Get |AnalogImpedance| reference load value, in Ohms.

        Returns:
            float: The reference load, in Ohms.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_reference = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceReferenceGet(self.hdwf, c_reference)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        reference = c_reference.value
        return reference

    def frequencySet(self, frequency: float) -> None:
        '''Set |AnalogImpedance| source frequency, in Hz.

        Parameters:
            frequency (float): The source frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceFrequencySet(self.hdwf, frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def frequencyGet(self) -> float:
        '''Get |AnalogImpedance| source frequency, in Hz.

        Returns:
            float: The source frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_frequency = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceFrequencyGet(self.hdwf, c_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency = c_frequency.value
        return frequency

    def amplitudeSet(self, amplitude: float) -> None:
        '''Set |AnalogImpedance| source amplitude value, in Volts.

        Parameters:
            amplitude (float): The source amplitude, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceAmplitudeSet(self.hdwf, amplitude)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def amplitudeGet(self) -> float:
        '''Get |AnalogImpedance| source amplitude, in Volts.

        Returns:
            float: The source amplitude, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_amplitude = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceAmplitudeGet(self.hdwf, c_amplitude)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        amplitude = c_amplitude.value
        return amplitude

    def offsetSet(self, offset: float) -> None:
        '''Set |AnalogImpedance| source offset, in Volts.

        Parameters:
            offset (float): The source offset, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceOffsetSet(self.hdwf, offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def offsetGet(self) -> float:
        '''Get |AnalogImpedance| source offset, in Volts.

        Returns:
            float: The source offset, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_offset = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceOffsetGet(self.hdwf, c_offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        offset = c_offset.value
        return offset

    def probeSet(self, resistance: float, capacitance: float) -> None:
        '''Set |AnalogImpedance| probe parameters.

        Parameters:
            resistance (float): The probe resistance, in Ohms.
            capacitance (float): The probe capacitance, in Farads.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceProbeSet(self.hdwf, resistance, capacitance)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def probeGet(self) -> Tuple[float, float]:
        '''Get |AnalogImpedance| probe parameters.

        Returns:
            Tuple[float, float]:
                A two-element tuple: the probe resistance, in Ohms, and the probe capacity, in Farads.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_resistance = typespec_ctypes.c_double()
        c_capacity = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceProbeGet(self.hdwf, c_resistance, c_capacity)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        resistance = c_resistance.value
        capacity = c_capacity.value
        return (resistance, capacity)

    def periodSet(self, period: int) -> None:
        '''Set the |AnalogImpedance| measurement period.

        Todo:
            Figure out what this setting is for, why it's an *int*, and what its physical unit is.

        Parameters:
            period (int): The measurement period.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedancePeriodSet(self.hdwf, period)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def periodGet(self) -> int:
        '''Get the |AnalogImpedance| measurement period.

        Todo:
            Figure out what this setting is for, why it's an *int*, and what its physical unit is.

        Returns:
            int: The measurement period.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_period = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogImpedancePeriodGet(self.hdwf, c_period)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        period = c_period.value
        return period

    def compReset(self) -> None:
        '''Reset the |AnalogImpedance| measurement computation.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceCompReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def compSet(self, open_resistance: float, open_reactance: float,
                short_resistance: float, short_reactance: float) -> None:
        '''Set the |AnalogImpedance| measurement computation parameters.

        Parameters:
            open_resistance (float): The open-circuit resistance, in Ohms.
            open_reactance (float): The open-circuit reactance, in Ohms.
            short_resistance (float): The short-circuit resistance, in Ohms.
            short_reactance (float): The short-circuit reactance, in Ohms.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogImpedanceCompSet(
            self.hdwf,
            open_resistance,
            open_reactance,
            short_resistance,
            short_reactance)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def compGet(self) -> Tuple[float, float, float, float]:
        '''Get the |AnalogImpedance| measurement computation parameters.

        Returns:
            Tuple[float, float, float, float]: The open-circuit resistance, open-circuit reactance,
            short-circuit resistance, and short-circuit reactance (all in Ohms).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_open_resistance = typespec_ctypes.c_double()
        c_open_reactance = typespec_ctypes.c_double()
        c_short_resistance = typespec_ctypes.c_double()
        c_short_reactance = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceCompGet(
            self.hdwf,
            c_open_resistance,
            c_open_reactance,
            c_short_resistance,
            c_short_reactance)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        open_resistance = c_open_resistance.value
        open_reactance = c_open_reactance.value
        short_resistance = c_short_resistance.value
        short_reactance = c_short_reactance.value
        return (open_resistance, open_reactance, short_resistance, short_reactance)

    def statusInput(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogImpedance| input measurement status.

        Parameters:
            channel_index(int): The channel for which to get gain and phase information.

        Returns:
            Tuple[float, float]: The gain and phase (in radians) of the current measurement.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_gain = typespec_ctypes.c_double()
        c_radian = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceStatusInput(
            self.hdwf,
            channel_index,
            c_gain,
            c_radian)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        gain = c_gain.value
        radian = c_radian.value
        return (gain, radian)

    def statusWarning(self, channel_index: int) -> int:
        '''Get warning for scope input range exceeded.

        Parameters:
            channel_index(int): The channel for which to get scope input range warning information.

        Returns:
            int: The warning, if any. 1 means *low*, 2 means *high*, 3 means *both*.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_warning = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogImpedanceStatusWarning(
            self.hdwf,
            channel_index,
            c_warning)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        warning = c_warning.value
        return warning

    def statusMeasure(self, measure: DwfAnalogImpedance) -> float:
        '''Retrieve the |AnalogImpedance| measurement status value.

        Parameters:
            measure (DwfAnalogImpedance): The quantity to measure.

        Returns:
            float: The value measured for the requested quantity.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_value = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogImpedanceStatusMeasure(self.hdwf, measure.value, c_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        value = c_value.value
        return value
