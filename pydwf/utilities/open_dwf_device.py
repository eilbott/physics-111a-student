'''Implementation of the |pydwf.utilities.openDwfDevice| function.'''

from typing import Optional, Callable, Dict, Any

from pydwf.core.dwf_library import DwfLibrary
from pydwf.core.dwf_device import DwfDevice
from pydwf.core.auxiliary.enum_types import DwfEnumConfigInfo, DwfEnumFilter
from pydwf.core.auxiliary.exceptions import PyDwfError


def _device_is_match(dwf: DwfLibrary, device_index: int, serial_number_filter: Optional[str],
                     device_id_filter: Optional[int], device_version_filter: Optional[int]) -> bool:

    if serial_number_filter is not None:
        if not dwf.deviceEnum.serialNumber(device_index).casefold().endswith(serial_number_filter.casefold()):
            return False

    if device_id_filter is not None or device_version_filter is not None:

        (device_id, device_version) = dwf.deviceEnum.deviceType(device_index)

        if device_id_filter is not None:
            if device_id != device_id_filter:
                return False

        if device_version_filter is not None:
            if device_version != device_version_filter:
                return False

    # The device passes all filters.
    return True


def openDwfDevice(
            dwf: DwfLibrary,
            enum_filter: Optional[DwfEnumFilter] = None,
            serial_number_filter: Optional[str] = None,
            device_id_filter: Optional[int] = None,
            device_version_filter: Optional[int] = None,
            score_func: Optional[Callable[[Dict[DwfEnumConfigInfo, Any]], Optional[Any]]] = None
        ) -> DwfDevice:
    '''Open a device identified by its serial number, optionally selecting a preferred configuration.

    This is a three-step process:

    1. The first step this function performs is to select a device for opening.

       To do this, device enumeration is performed, resulting in a list of all reachable Digilent Waveforms devices.

       For this initial enumeration process the *enum_filter* parameter can be used to only list certain types of
       devices (for example, only Analog Discovery 2 devices, or Digital Discovery devices).
       If omitted (the default), all Digilent Waveforms devices will be listed.

       Then, if the *serial_number_filter* parameter is given, the list will be filtered to exclude devices whose
       serial numbers do not match.

       If the list that remains has a single device, this device will be used. If not, a |PyDwfError:link| is raised.

    2. The next step is to select a |device configuration:link| for the selected device.

       For many use cases, the default configuration that provides a balanced tradeoff works fine.
       If no *score_func* is provided, this default configuration will be used.

       If a *score_func* parameter is provided, it should be a function (or lambda expression) that
       takes a single parameter *configuration_info*, which is a dictionary with |DwfEnumConfigInfo:link| keys,
       and parameters values for a specific device configuration.

       The *score_func* should return None if the configuration is entirely unsuitable, or otherwise a score that
       reflects the suitability of that particular configuration for the task at hand.

       The *openDwfDevice* method will go through all available device configurations, construct a dictionary
       of all parameters that describe the configuration, call the *score_func* with that dictionary as a parameter,
       and examine the score value it returns. If multiple suitable device configurations are found (i.e., the
       *score_func* does not return None), it will select the configuration with the highest score.

       This may all sounds pretty complicated, but in practice this parameter is quite easy to define for most
       common use-cases.

       As an example, to select a configuration that maximizes the analog input buffer size, simply use this:

       .. code-block:: python

          from pydwf import DwfEnumConfigInfo
          from pydwf.utilities import openDwfDevice

          def maximize_analog_in_buffer_size(config_parameters):
              return config_parameters[DwfEnumConfigInfo.AnalogInBufferSize]

          with openDwfDevice(dwf, score_func = maximize_analog_in_buffer_size) as device:
              use_device_with_big_analog_in_buffer(device)

    3. As a final step, the selected device is opened using the selected device configuration, and the
       newly instantiated |DwfDevice:link| instance is returned.

    Note:
        This method can take several seconds to complete. This long duration is caused by the use
        of the |DeviceEnum.enumerateDevices:link| method.

    Parameters:
        dwf (DwfLibrary): The |DwfLibrary:link| used to open the device.
        enum_filter (Optional[DwfEnumFilter]):
            An optional filter to limit the device enumeration to certain device types.
            If None, enumerate all devices.
        serial_number_filter (str): The serial number filter used to select a specific device.
            A device is considered to match if its 12-digit serial number ends with the serial number filter string
            (case is ignored).
        device_id_filter (int): The device ID filter to use.
        device_version_filter (int): The device version filter to use.
        score_func (Optional[Callable[[Dict[DwfEnumConfigInfo, Any]], Optional[Any]]]):
            A function to score a configuration of the selected device.
            See the description above for details.

    Returns:
        DwfDevice: The |DwfDevice:link| created as a result of this call.

    Raises:
        PyDwfError: could not select a unique candidate device, or no usable configuration detected.
    '''

    if enum_filter is None:
        enum_filter = DwfEnumFilter.All

    # The dwf.deviceEnum.enumerateDevices() function builds an internal table of connected devices
    # that can be queried using other dwf.deviceEnum methods.
    num_devices = dwf.deviceEnum.enumerateDevices(enum_filter)

    if num_devices == 0:
        raise PyDwfError(f'No Digilent Waveforms device found in device class "{enum_filter.name}".')

    # Check all filters and keep the devices that match all specified criteria.
    candidate_devices = [
        device_index for device_index in range(num_devices)
        if _device_is_match(dwf, device_index, serial_number_filter, device_id_filter, device_version_filter)
    ]

    if len(candidate_devices) == 0:
        raise PyDwfError('No matching Digilent Waveforms devices found after applying filters.')

    if len(candidate_devices) > 1:
        # Multiple devices found after applying the given filters.
        raise PyDwfError('Multiple matching Digilent Waveforms devices found after applying filters.')

    # We now have a single candidate device left.
    device_index = candidate_devices[0]

    if score_func is None:
        # If no 'score_func' was specified, just open the device with the default (first) configuration.
        return dwf.deviceControl.open(device_index)

    # The caller specified a 'score_func'.
    # We will examine all configuration and pick the one that has the highest score.

    num_config = dwf.deviceEnum.enumerateConfigurations(device_index)

    best_configuration_index = None
    best_configuration_score = None

    for configuration_index in range(num_config):

        configuration_info = {
            configuration_parameter:
                dwf.deviceEnum.configInfo(configuration_index, configuration_parameter)
                for configuration_parameter in DwfEnumConfigInfo
        }

        configuration_score = score_func(configuration_info)

        if configuration_score is None:
            continue

        if best_configuration_index is None or configuration_score > best_configuration_score:
            best_configuration_index = configuration_index
            best_configuration_score = configuration_score

    if best_configuration_index is None:
        raise PyDwfError('No acceptable configuration was found.')

    return dwf.deviceControl.open(device_index, best_configuration_index)
