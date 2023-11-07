#! /usr/bin/env python3

"""Demo of the AnalogIn "Record" acquisition mode.

Summary
-------

This demo generates signals on two AnalogOut instrument channels and captures them on two AnalogIn instrument
channels, displaying the result as a graph using matplotlib.

Description
-----------

When using the AnalogIn instrument with the "Record" acquisition mode, we prepare the analog input channel and
(if desired) trigger settings, then start the acquisition using a call to analogIn.configure().

Next, we enter a loop where we continuously fetch data from the instrument by calling analogIn.status(True).
This is repeated until analogIn.status() returns DwfState.Done. Note that this last status() call also transfers
acquisition data that needs to be processed.

After each status() call, we get information on the acquisition status by calling statusRecord(). This call
returns three numbers: counts of available, lost, and corrupted samples.

For perfect acquisition, the lost and corrupt counts should be zero. If the acquisition requires more bandwidth than
can be accommodated on the USB-2 link to the device (i.e., the sample frequency is too high), or if we fetch data too
slowly from our user program, we may see non-zero lost and corrupted counts.

If this happens, the documentation provides no guidance on handling this other than to suggest that the acquisition
sample rate should be lowered, and/or the process should fetch data  more quickly. The examples provided by Digilent
suggest that the way to handle nonzero "lost" samples is to skip over them; this is what we do in the program below
(filling the lost samples with NaNs).

Assuming the lost and corrupt counts are zero, the 'available' count gives the number of valid samples available in
the local (PC-side) buffer. These samples can be obtained using calls to statusData(), statusData2(), or
statusData16(). The pydwf library implements these functions by having them allocate a sufficiently-sized local
numpy array, reading the sample data into it, and returning that array.

At the end of the acquisition, i.e., after the status() function returns DwfState.Done, these sub-arrays are
concatenated to deliver the full sample record of the acquisition.

At that point, we discard all but the last (record_length * sample_frequency) samples that constitute the requested
recording length. The preceding samples were received from the device, but the first few samples may be garbled,
and the total number of samples received will generally exceed the number of samples requested, sometimes by a
considerable margin.

The discarding process is also needed to make sure that the trigger position is in a predictable and reproducible
place. After discarding the first samples of the acquisition to get the requested length, the first remaining
sample is at the time, measured in seconds, returned by the analogIn.triggerPositionStatus() call, relative to
the trigger moment.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from pydwf import (
    DwfLibrary,
    DwfEnumConfigInfo,
    DwfAnalogOutNode,
    DwfAnalogOutFunction,
    DwfAcquisitionMode,
    DwfTriggerSource,
    DwfAnalogInTriggerType,
    DwfTriggerSlope,
    DwfState,
    DwfAnalogInFilter,
    PyDwfError,
)
from pydwf.utilities import openDwfDevice

# how often the scope samples the input signal

# TODO 10.2 part c: edit these so that they are the correct values
SINE_FREQUENCY = 0
ANALOG_OUT_AMPLITUDE = 0
ANALOG_OUT_OFFSET = 0


SAMPLE_FREQUENCY = 100e3
# We want to see 5 full cycles in the acquisition window.
RECORD_LENGTH = (
    5 / SINE_FREQUENCY
)  # this is how long the scope records for (until we display the data it records and iterate again through the while loop)
# TODO 10.3 b - set this to true
USE_TRIGGER = False


def configure_analog_output(
    analogOut, analog_out_frequency, analog_out_amplitude, analog_out_offset
):
    CH1 = 0

    node = DwfAnalogOutNode.Carrier

    analogOut.reset(-1)  # Reset both channels.
    analogOut.nodeEnableSet(CH1, node, True)

    # TODO 10.2 part c: write code to configure the right signal for the wavegen to output.
    # ANSWER:

    # END ANSWER

    # Start output on channel 1.
    analogOut.configure(CH1, True)


def configure_trigger(analogIn, trigger_flag, record_length, channel, trigger_level):
    if trigger_flag:
        # Position of first sample relative to the trigger.
        # Setting it to -0.5 * record_length puts the trigger halfway the capture window.
        trigger_position = -0.5 * record_length
        # Set up trigger for the analog input instrument.
        # We will trigger on the rising transitions of CH1 (the "cosine" channel) through 0V.
        analogIn.triggerSourceSet(DwfTriggerSource.DetectorAnalogIn)
        analogIn.triggerChannelSet(
            channel
        )  # You trigger when analog 2 rises UP (This is why CH 1 needs to be plugged in)
        analogIn.triggerTypeSet(DwfAnalogInTriggerType.Edge)
        analogIn.triggerConditionSet(DwfTriggerSlope.Rise)
        analogIn.triggerPositionSet(trigger_position)
        analogIn.triggerLevelSet(trigger_level)
        analogIn.triggerHysteresisSet(
            0.10
        )  # A small amount of hysteresis to make sure we only see rising edges.


def acquire_one_round_samps(analogIn, samples, channels, trigger_flag, acquisition_nr):
    total_samples_lost = total_samples_corrupted = 0
    while True:
        status = analogIn.status(True)
        (
            current_samples_available,
            current_samples_lost,
            current_samples_corrupted,
        ) = analogIn.statusRecord()

        total_samples_lost += current_samples_lost
        total_samples_corrupted += current_samples_corrupted

        if current_samples_lost != 0:
            # Append NaN samples as placeholders for lost samples.
            # This follows the Digilent example.
            # We haven't verified yet that this is the proper way to handle lost samples.
            lost_samples = np.full((current_samples_lost, 2), np.nan)
            samples.append(lost_samples)

        if current_samples_available != 0:
            # Append samples read from both channels.
            # Note that we read the samples separately for each channel;
            # We then put them into the same 2D array with shape (current_samples_available, 2).
            current_samples = np.vstack(
                [
                    analogIn.statusData(channel_index, current_samples_available)
                    for channel_index in channels
                ]
            ).transpose()
            samples.append(current_samples)

        if status == DwfState.Done:
            # We received the last of the record samples.
            # Note the time, in seconds, of the first valid sample, and break from the acquisition loop.
            if trigger_flag:
                time_of_first_sample = analogIn.triggerPositionStatus()
            else:
                time_of_first_sample = 0.0
            break

    if total_samples_lost != 0:
        print(
            "[{}] - WARNING - {} samples were lost! Reduce sample frequency.".format(
                acquisition_nr, total_samples_lost
            )
        )

    if total_samples_corrupted != 0:
        print(
            "[{}] - WARNING - {} samples could be corrupted! Reduce sample frequency.".format(
                acquisition_nr, total_samples_corrupted
            )
        )
    return time_of_first_sample


def run_demo(
    analogIn,
    sample_frequency,
    record_length,
    trigger_flag,
    signal_frequency,
    signal_amplitude,
):
    """Configure the analog input, and perform repeated acquisitions and present them graphically."""
    # pylint: disable=too-many-statements,too-many-branches, too-many-locals

    # Configure analog input instrument acquisition.

    CH1 = 0

    channels = (CH1,)

    for channel_index in channels:
        analogIn.channelEnableSet(channel_index, True)  # Enables the scope channel
        analogIn.channelFilterSet(channel_index, DwfAnalogInFilter.Average)
        # analogIn.channelRangeSet(channel_index, 5.0)  # Range of scope

    analogIn.acquisitionModeSet(DwfAcquisitionMode.Record)  # Set more to record
    analogIn.frequencySet(
        sample_frequency
    )  # How often are you gonna sample from signal?
    analogIn.recordLengthSet(record_length)  # How long to record before graph updates

    # Trigger level, in Volts.
    trigger_level = 0
    configure_trigger(analogIn, trigger_flag, record_length, CH1, trigger_level)

    # Calculate number of samples for each acquisition.
    num_samples = round(sample_frequency * record_length)

    # Outer loop: perform repeated acquisitions.
    acquisition_nr = 0

    ch1_line = None
    # 9.1.4: Uncomment the line below:
    # average = None
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 8))
    plt.subplots_adjust(hspace=0.5)
    while True:
        acquisition_nr += 1  # Increment acquisition number.

        print("[{}] Recording {} samples ...".format(acquisition_nr, num_samples))

        # Inner loop: single acquisition, receive data from AnalogIn instrument and display it.

        samples = []

        analogIn.configure(False, True)  # Start acquisition sequence.

        time_of_first_sample = acquire_one_round_samps(
            analogIn, samples, channels, trigger_flag, acquisition_nr
        )

        samples = np.concatenate(samples)

        if len(samples) > num_samples:
            discard_count = len(samples) - num_samples
            print(
                "[{}] - NOTE - discarding oldest {} of {} samples ({:.1f}%); keeping {} samples.".format(
                    acquisition_nr,
                    discard_count,
                    len(samples),
                    100.0 * discard_count / len(samples),
                    num_samples,
                )
            )

            samples = samples[discard_count:]

        ch1_line = plot_samples(
            ch1_line,
            samples,
            sample_frequency,
            signal_frequency,
            time_of_first_sample,
            acquisition_nr,
            num_samples,
            record_length,
            trigger_flag,
            trigger_level,
            ax0,
        )

        # TODO 10.4: uncomment this line Now, plot the FFT on the second subplot
        # plot_fft(samples[:, CH1], sample_frequency, ax1)
        plt.pause(1e-3)
        if len(plt.get_fignums()) == 0:
            break


def plot_fft(samples, sample_frequency, ax1):
    # 10.4 TODO: get the FFT Frquency (fft_freq) and magnitude (ffr_magnitude)
    # Hint: use np.fft.fft on samples. the first half of the result is the positive frequencies. You'll need to double these
    # (except for 0) due to symmetry of the negative frequencies.
    fft_freq = fft_magnitude = None
    # END TODO

    # Plotting the FFT on the provided axis
    ax1.plot(fft_freq, fft_magnitude)
    ax1.set_title("FFT of the signal")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude")
    # 10.4 TODO: adjust the x-limit to see the frequencies you want
    ax1.set_xlim(0, sample_frequency / 2)
    # ax1.set_xlim(0, 20)
    ax1.set_ylim(0)  # Adjust if needed


def plot_samples(
    ch1_line,
    samples,
    sample_frequency,
    signal_frequency,
    time_of_first_sample,
    acquisition_nr,
    num_samples,
    record_length,
    trigger_flag,
    trigger_level,
    ax0,
):
    CH1 = 0

    # Subplot 1: Analog Input

    t = time_of_first_sample + np.arange(len(samples)) / sample_frequency

    if ch1_line is None:
        # This is the first time we plot acquisition data.
        ax0.grid()
        ax0.set_title(
            "AnalogIn acquisition #{}\n{} samples ({} seconds at {} Hz)\nsignal frequency: {} Hz".format(
                acquisition_nr,
                num_samples,
                record_length,
                sample_frequency,
                signal_frequency,
            )
        )

        if trigger_flag:
            ax0.set_xlabel(
                "time relative to trigger [s]\ntriggering on rising zero transition of channel 2"
            )
        else:
            ax0.set_xlabel("acquisition time [s]")

        ax0.set_ylabel("signal [V]")

        plt.xlim(-0.05 * record_length, 1.05 * record_length)

        ax0.set_ylim(-7, 7)
        # 9.1.3: uncomment the line below and comment out the line above
        # plt.ylim(-1.1 * signal_amplitude + 1.9, +1.1 * signal_amplitude + 1.9)

        if trigger_flag:
            ax0.axvline(0.0, c="r")
            ax0.axhline(trigger_level, c="r")

        (ch1_line,) = ax0.plot(
            t, samples[:, CH1], color="#346f9f", label="channel 1 (cos)"
        )

        ax0.legend(loc="upper right")
    else:
        # The plot is already available. Just update the acquisition data.
        ch1_line.set_xdata(t)
        ch1_line.set_ydata(samples[:, CH1])

    return ch1_line


def main():
    dwf = DwfLibrary()

    def maximize_analog_in_buffer_size(configuration_parameters):
        """Select the configuration with the highest possible analog in buffer size."""
        return configuration_parameters[DwfEnumConfigInfo.AnalogInBufferSize]

    try:
        with openDwfDevice(
            dwf,
            score_func=maximize_analog_in_buffer_size,
        ) as device:
            analogOut = device.analogOut
            analogIn = device.analogIn

            # Signal amplitude in Volt.
            # The AnalogOut instrument can do 10 Vpp centered around 0 V.
            # However, we use the AnalogIn instrument with a ~ 5 Vpp range centered around 0 V,
            #   so for our example we set the analog output signal amplitude to 2.5 V.

            # Signal offset in Volt.

            print(
                "Configuring analog output signals ({} Hz) ...".format(SINE_FREQUENCY)
            )

            configure_analog_output(
                analogOut, SINE_FREQUENCY, ANALOG_OUT_AMPLITUDE, ANALOG_OUT_OFFSET
            )
            print("starting analog output:")
            # 10.3: TODO: remove this while loop, as the scope logic will have an infinite while loop to keep the program running.
            # Also uncomment run_demo function call

            while True:
                time.sleep(
                    2.0
                )  # Wait for a bit to ensure the stability of the analog output signals.

            # run_demo(
            #     analogIn,
            #     SAMPLE_FREQUENCY,
            #     RECORD_LENGTH,
            #     USE_TRIGGER,
            #     SINE_FREQUENCY,
            #     ANALOG_OUT_AMPLITUDE,
            # )

    except PyDwfError as exception:
        print("PyDwfError:", exception)


if __name__ == "__main__":
    main()
