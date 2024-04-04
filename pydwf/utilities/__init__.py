'''The |pydwf.utilities| package provides utility functions for managing Digilent Waveforms devices.

It provides high-level functions that reflect best-practice implementations for common use-cases of |pydwf|.

Currently, it only provides the `openDwfDevice` function, as well as functionality to configure multiple parameters
of complex instruments.
'''

# Enumerate the functions that are directly available from the pydwf.utilities package:

from pydwf.utilities.open_dwf_device import openDwfDevice

from pydwf.utilities.configuration import (
        configure_analog_in_instrument,
        configure_analog_in_channel,
        configure_analog_out_channel,
        configure_analog_out_channel_node,
        configure_digital_in_instrument,
        configure_digital_out_instrument,
        configure_digital_out_channel
    )
