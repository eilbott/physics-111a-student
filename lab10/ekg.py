#! /usr/bin/env python3
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
SAMPLE_FREQUENCY = 100e3
RECORD_LENGTH = 0.05  # this is how long the scope records for (until we display the data it records and iterate again through the while loop)

USE_TRIGGER = True
# We want to see 5 full cycles in the acquisition window.
SINE_FREQUENCY = 2000
ANALOG_OUT_AMPLITUDE = 0.5
ANALOG_OUT_OFFSET = 1.9


def configure_analog_output(
    analogOut, analog_out_frequency, analog_out_amplitude, analog_out_offset
):
    CH1 = 0

    node = DwfAnalogOutNode.Carrier

    analogOut.reset(-1)  # Reset both channels.
    analogOut.nodeEnableSet(CH1, node, True)

    # 9.1.1: (DC voltage to lit up LED) uncomment the relevant lines and edit where necessary:
    # 9.1.2: uncomment below and comment above
    analogOut.nodeFunctionSet(CH1, node, DwfAnalogOutFunction.Sine)
    analogOut.nodeFrequencySet(CH1, node, analog_out_frequency)
    analogOut.nodeAmplitudeSet(CH1, node, analog_out_amplitude)
    analogOut.nodeOffsetSet(CH1, node, analog_out_offset)

    # Start output on channel 1.
    analogOut.configure(CH1, True)


def configure_trigger(analogIn, trigger_flag, record_length, channel, trigger_level):
    if trigger_flag:
        # Position of first sample relative to the trigger.
        # Setting it to -0.5 * record_length puts the trigger halfway the capture window.
        trigger_position = -0.5 * record_length
        # Set up trigger for the analog input instrument.
        # We will trigger on the rising transitions of CH2 (the "cosine" channel) through 0V.
        analogIn.triggerSourceSet(DwfTriggerSource.AnalogOut1)
        analogIn.triggerChannelSet(
            channel
        )  # You trigger when analog 2 rises UP (This is why CH 2 needs to be plugged in)
        analogIn.triggerTypeSet(DwfAnalogInTriggerType.Edge)
        analogIn.triggerConditionSet(DwfTriggerSlope.Rise)
        analogIn.triggerPositionSet(trigger_position)
        analogIn.triggerLevelSet(trigger_level)
        analogIn.triggerHysteresisSet(
            0.010
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
            # print("PREEEE MAXXXXX", np.max(current_samples))
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


def get_lock_in(samples, lock_in_vals, lock_in_times, time_point):
    # Assuming that the driving signal is in column 0 and EKG is in column 1
    driving_signal = samples[:, 0]
    ekg_signal = samples[:, 1]
    N = len(driving_signal)

    # Phase information from the driving signal
    analytic_signal = np.fft.ifft(np.fft.fft(driving_signal) * 2)
    phase = np.angle(analytic_signal)

    phase = 0
    # Construct a sine wave using phase information
    time_2 = np.arange(N) / SINE_FREQUENCY
    ref_sig = 0.25 * np.sin(2 * np.pi * time_2 + phase) + 1.9
    ref_cos = 0.25 * np.sin(2 * np.pi * time_2 + phase + np.pi / 2) + 1.9

    curr_lock = np.sqrt(
        (np.sum(ref_sig * ekg_signal) / N) ** 2
        + (np.sum(ref_cos * ekg_signal) / N) ** 2
    )
    if len(lock_in_vals) >= 300:
        lock_in_vals.pop(0)
        lock_in_times.pop(0)

    lock_in_vals.append(curr_lock)
    lock_in_times.append(time_point)


from scipy.fft import fft, fftfreq
from scipy.signal import blackmanharris


def lockin_fft(lock_in_times, lock_in_vals):
    # Calculate the difference between timestamps to get the sample spacing
    # We'll assume that the sampling is evenly spaced
    fs = 1.0 / np.mean(np.diff(lock_in_times))

    # Window function
    N = len(lock_in_vals)
    w = blackmanharris(N)

    # FFT
    yf = fft(lock_in_vals * w)
    xf = fftfreq(N, 1 / fs)

    # Since we are interested in the positive frequencies from 0 to 4 Hz,
    # we only need to consider the first half of the FFT output
    mask = (xf >= 0.3) & (xf <= 4)
    xf = xf[mask]
    yf = np.abs(yf[mask])

    return xf, yf


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
    CH2 = 1

    channels = (CH1, CH2)

    for channel_index in channels:
        analogIn.channelEnableSet(channel_index, True)  # Enables the scope channel
        analogIn.channelFilterSet(channel_index, DwfAnalogInFilter.Average)
        analogIn.channelRangeSet(channel_index, 24.0)  # Range of scope

    analogIn.acquisitionModeSet(DwfAcquisitionMode.Record)  # Set more to record
    analogIn.frequencySet(
        sample_frequency
    )  # How often are you gonna sample from signal?
    analogIn.recordLengthSet(record_length)  # How long to record before graph updates

    # Trigger level, in Volts.
    trigger_level = 0.0
    configure_trigger(analogIn, trigger_flag, record_length, CH1, trigger_level)

    # Calculate number of samples for each acquisition.
    num_samples = round(sample_frequency * record_length)

    # Outer loop: perform repeated acquisitions.
    acquisition_nr = 0

    ch1_line = ch2_line = None
    lock_in_ch = fft_ch = None
    lock_in_vals = []
    lock_in_times = []
    first_time = time.time()
    xf = yf = None

    # Create subplots
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 8))
    while True:
        acquisition_nr += 1  # Increment acquisition number.

        print("[{}] Recording {} samples ...".format(acquisition_nr, num_samples))

        # Inner loop: single acquisition, receive data from AnalogIn instrument and display it.

        samples = []

        analogIn.configure(False, True)  # Start acquisition sequence.

        time_of_first_sample = acquire_one_round_samps(
            analogIn, samples, channels, trigger_flag, acquisition_nr
        )
        acquire_time = time.time()

        # Concatenate all acquired samples. The result is an (n, 2) array of sample values.
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

        get_lock_in(samples, lock_in_vals, lock_in_times, acquire_time - first_time)
        if len(lock_in_vals) >= 40 and acquisition_nr % 20 == 0:
            xf, yf = lockin_fft(lock_in_times, lock_in_vals)

        # want a changing plot of x = lock_in_times, y = lock_in_vals right here. it should update every while loop

        # want a constantly updating plot of xf, yf, right here. should only be plotted after len(lock_in_vals >= 40)

        ch1_line, ch2_line = plot_samples(
            ch1_line,
            ch2_line,
            samples,
            sample_frequency,
            signal_frequency,
            time_of_first_sample,
            acquisition_nr,
            num_samples,
            record_length,
            trigger_flag,
            trigger_level,
            signal_amplitude,
            ax0,
        )
        lock_in_ch = plot_lockin(ax1, lock_in_ch, lock_in_times, lock_in_vals)
        fft_ch = plot_fft(ax2, fft_ch, xf, yf)

        if len(plt.get_fignums()) == 0:
            # User has closed the window, finish.
            # 9.1.4: Uncomment the line below to save the file to the home directory:
            # np.save("filename.npy", average)
            break


def plot_samples(
    ch1_line,
    ch2_line,
    samples,
    sample_frequency,
    signal_frequency,
    time_of_first_sample,
    acquisition_nr,
    num_samples,
    record_length,
    trigger_flag,
    trigger_level,
    signal_amplitude,
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

        if trigger_flag:
            ax0.set_xlim(-0.1 * record_length, 0.1 * record_length)
        else:
            ax0.set_xlim(-0.05 * record_length, 1.05 * record_length)

        ax0.set_ylim(-12, 12)
        # 9.1.3: uncomment the line below and comment out the line above
        # plt.ylim(-1.1 * signal_amplitude + 1.9, +1.1 * signal_amplitude + 1.9)

        if trigger_flag:
            ax0.axvline(0.0, c="r")
            ax0.axhline(trigger_level, c="r")

        (ch1_line,) = ax0.plot(
            t, samples[:, CH1], color="#346f9f", label="channel 1 (cos)"
        )
        (ch2_line,) = ax0.plot(
            t, samples[:, 1], color="#ffdd56", label="channel 2 (sin)"
        )

        ax0.legend(loc="upper right")
    else:
        # The plot is already available. Just update the acquisition data.
        ch1_line.set_xdata(t)
        ch1_line.set_ydata(samples[:, CH1])

        ch2_line.set_xdata(t)
        ch2_line.set_ydata(samples[:, 1])

    plt.pause(1e-3)
    return ch1_line, ch2_line


def plot_lockin(ax1, lock_in_ch, lock_in_times, lock_in_vals):
    if lock_in_ch:
        lock_in_ch.set_xdata(lock_in_times)
        lock_in_ch.set_ydata(lock_in_vals)
        ax1.set_ylim(min(lock_in_vals), max(lock_in_vals))
        ax1.set_xlim(min(lock_in_times), max(lock_in_times))

    else:
        ax1.grid()
        ax1.set_title("lock in")
        (lock_in_ch,) = ax1.plot(
            lock_in_times, lock_in_vals, color="#346f9f", label="channel 1 (cos)"
        )
        ax1.set_xlim(0, 40)
        ax1.set_ylim(min(lock_in_vals), max(lock_in_vals))
    return lock_in_ch


def plot_fft(ax2, fft_ch, xf, yf):
    if xf is None or yf is None:
        return None
    if fft_ch:
        fft_ch.set_xdata(xf)
        fft_ch.set_ydata(yf)
    else:
        ax2.grid()
        ax2.set_title("fft ")
        ax2.set_ylim(0, 2)
        (fft_ch,) = ax2.plot(xf, yf)
    return fft_ch


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
            time.sleep(
                2.0
            )  # Wait for a bit to ensure the stability of the analog output signals.

            run_demo(
                analogIn,
                SAMPLE_FREQUENCY,
                RECORD_LENGTH,
                USE_TRIGGER,
                SINE_FREQUENCY,
                ANALOG_OUT_AMPLITUDE,
            )

    except PyDwfError as exception:
        print("PyDwfError:", exception)


if __name__ == "__main__":
    main()
