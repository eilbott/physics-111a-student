'''The |pydwf.core.api.protocol_spi| module provides a single class: |ProtocolSPI|.'''

# pylint: disable=too-many-lines

from typing import List

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.enum_types import DwfDigitalOutIdle


class ProtocolSPI(AbstractDwfDeviceSubAPI):

    '''The |ProtocolSPI| class provides access to the SPI protocol functionality of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently accessible via its
        |protocol.spi:link| attribute.
    '''

    # pylint: disable=too-many-public-methods

    def reset(self) -> None:
        '''Reset the SPI protocol support.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalSpiReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def frequencySet(self, frequency: float) -> None:
        '''Set the SPI frequency, in Hz.

        Parameters:
            frequency (float): SPI frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiFrequencySet(self.hdwf, frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def clockSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) for the SPI clock signal.

        Parameters:
            channel_index (int):
                The digital channel (pin) where the SPI clock signal will be generated.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiClockSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def dataSet(self, spi_data_bit: int, channel_index: int) -> None:
        '''Set the digital channel (pin) for an SPI data bit.

        Parameters:
            spi_data_bit (int):

                The data bit to configure:

                * 0 — DQ0 / MOSI / SISO
                * 1 — DQ1 / MISO
                * 2 — DQ2
                * 3 — DQ3

            channel_index (int):
                The digital channel (pin) for this data bit.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiDataSet(self.hdwf, spi_data_bit, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def idleSet(self, spi_data_bit: int, idle_mode: DwfDigitalOutIdle) -> None:
        '''Set the idle behavior for an SPI data bit.

        Parameters:
            spi_data_bit (int):

                The data bit to configure:

                * 0 — DQ0 / MOSI / SISO
                * 1 — DQ1 / MISO
                * 2 — DQ2
                * 3 — DQ3

            idle_mode (DwfDigitalOutIdle):
                The idle behavior of this bit.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        result = self.lib.FDwfDigitalSpiIdleSet(self.hdwf, spi_data_bit, idle_mode.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def modeSet(self, spi_mode: int) -> None:
        '''Set the SPI mode.

        Parameters:
            spi_mode (int):

                The values for CPOL (polarity) and CPHA (phase) to use with the attached slave device:

                * 0 — CPOL = 0, CPHA = 0
                * 1 — CPOL = 0, CPHA = 1
                * 2 — CPOL = 1, CPHA = 0
                * 3 — CPOL = 1, CPHA = 1

                Refer to the slave device's datasheet to select the correct value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiModeSet(self.hdwf, spi_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def orderSet(self, bit_order: int) -> None:
        '''Set the SPI data bit order.

        Parameters:
            bit_order (int):

                Select the bit order of each word sent out:

                * 1 — MSB first, LSB last
                * 0 — LSB first, MSB last

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiOrderSet(self.hdwf, bit_order)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def delaySet(self, start: int, cmd: int, word: int, stop: int) -> None:
        '''Set the SPI delays.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiDelaySet(self.hdwf, start, cmd, word, stop)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def selectSet(self, channel: int, level: int) -> None:
        '''Set SPI device select channel and level.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiSelectSet(self.hdwf, channel, level)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def select(self, channel_index: int, level: int) -> None:
        '''Set the chip select (CS) status.

        Parameters:

            channel_index (int):
                The digital channel (pin) for the Chip Select signal.

            level (int):

                The Chip Select level to configure.

                *  0 — low
                *  1 — high
                * -1 — Z (high impedance)

                The CS (chip select) is an active-low signal, from the SPI bus master to a
                specific SPI slave device. Before starting a bus request, the master should
                set CS to 0 for the chip it wants to talk to.

                Each slave on an SPI bus has its own CS line. At most one of them should be
                selected at any time.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSpiSelect(self.hdwf, channel_index, level)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def writeRead(self, transfer_type: int, bits_per_word: int, tx: List[int]) -> List[int]:
        '''Write and read multiple SPI data-words, with up to 8 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…8).
            tx (List[int]):
                The data-words to write.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the write/read operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        tx_buffer = buffer_type(*tx_list)
        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiWriteRead(
            self.hdwf,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def writeRead16(self, transfer_type: int, bits_per_word: int, tx: List[int]) -> List[int]:
        '''Write and read multiple SPI data-words, with up to 16 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…16).
            tx (list[int]):
                The data-words to write.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the write/read operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_short * number_of_words

        tx_buffer = buffer_type(*tx_list)
        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiWriteRead16(
            self.hdwf,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def writeRead32(self, transfer_type: int, bits_per_word: int, tx: List[int]) -> List[int]:
        '''Write and read multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            tx (List[int]):
                The data-words to write.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the write/read operation.
        '''

        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_int * number_of_words

        tx_buffer = buffer_type(*tx_list)
        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiWriteRead32(
            self.hdwf,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def read(self, transfer_type: int, bits_per_word: int, number_of_words: int) -> List[int]:
        '''Read multiple SPI data-words, with up to 8 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…8).
            number_of_words (int):
                The number of data-words to read.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiRead(
            self.hdwf,
            transfer_type,
            bits_per_word,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def readOne(self, transfer_type: int, bits_per_word: int) -> int:
        '''Read a single SPI data-word, with up to 32 bits.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits of the data-word (1…32).

        Returns:
            int: The data-word received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''
        c_rx = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalSpiReadOne(self.hdwf, transfer_type, bits_per_word, c_rx)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        rx = c_rx.value
        return rx

    def read16(self, transfer_type: int, bits_per_word: int, number_of_words: int) -> List[int]:
        '''Read multiple SPI data-words, with up to 16 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…16).
            number_of_words (int):
                The number of data-words to read.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''

        buffer_type = typespec_ctypes.c_unsigned_short * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiRead16(
            self.hdwf,
            transfer_type,
            bits_per_word,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def read32(self, transfer_type: int, bits_per_word: int, number_of_words: int) -> List[int]:
        '''Read multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            number_of_words (int):
                The number of data-words to read.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''

        buffer_type = typespec_ctypes.c_unsigned_int * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiRead32(
            self.hdwf,
            transfer_type,
            bits_per_word,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def write(self, transfer_type: int, bits_per_word: int, tx: List[int]) -> None:
        '''Write multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            tx (List[int]):
                The data-words to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalSpiWrite(
            self.hdwf,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def writeOne(self, transfer_type: int, bits_per_word: int, tx: int) -> None:
        '''Write a single SPI data-word, with up to 32 bits.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits of the data-word (1…32).
            tx (int):
                The data-word to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        result = self.lib.FDwfDigitalSpiWriteOne(self.hdwf, transfer_type, bits_per_word, tx)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def write16(self, transfer_type: int, bits_per_word: int, tx: List[int]) -> None:
        '''Write multiple SPI data-words, with up to 16 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…16).
            tx (List[int]):
                The data-words to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_short * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalSpiWrite16(
            self.hdwf,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def write32(self, transfer_type: int, bits_per_word: int, tx: List[int]) -> None:
        '''Write multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            tx (List[int]):
                The data-words to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_int * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalSpiWrite32(
            self.hdwf,
            transfer_type,
            bits_per_word,
            tx_buffer, number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def cmdWriteRead(self, command_bits: int, command_value: int, dummy_bits: int,
                     transfer_type: int, bits_per_word: int, tx: List[int]) -> List[int]:
        '''Send command and write and read multiple SPI data-words, with up to 8 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…8).
            tx (List[int]):
                The data-words to write.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the write/read operation.
        '''

        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        tx_buffer = buffer_type(*tx_list)
        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiCmdWriteRead(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def cmdWriteRead16(self, command_bits: int, command_value: int, dummy_bits: int,
                       transfer_type: int, bits_per_word: int, tx: List[int]) -> List[int]:
        '''Send command and write and read multiple SPI data-words, with up to 16 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…16).
            tx (list[int]):
                The data-words to write.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the write/read operation.
        '''

        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_short * number_of_words

        tx_buffer = buffer_type(*tx_list)
        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiCmdWriteRead16(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def cmdWriteRead32(self, command_bits: int, command_value: int, dummy_bits: int,
                       transfer_type: int, bits_per_word: int, tx: List[int]) -> List[int]:
        '''Send command and write and read multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            tx (List[int]):
                The data-words to write.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the write/read operation.
        '''

        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_int * number_of_words

        tx_buffer = buffer_type(*tx_list)
        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiCmdWriteRead32(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def cmdRead(self, command_bits: int, command_value: int, dummy_bits: int,
                transfer_type: int, bits_per_word: int, number_of_words: int) -> List[int]:
        '''Send command and read multiple SPI data-words, with up to 8 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…8).
            number_of_words (int):
                The number of data-words to read.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiCmdRead(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def cmReadOne(self, command_bits: int, command_value: int, dummy_bits: int,
                  transfer_type: int, bits_per_word: int) -> int:
        '''Send command and read a single SPI data-word, with up to 32 bits.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits of the data-word (1…32).

        Returns:
            int: The data-word received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''
        c_rx = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalSpiCmdReadOne(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            c_rx)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        rx = c_rx.value
        return rx

    def cmdRead16(self, command_bits: int, command_value: int, dummy_bits: int,
                  transfer_type: int, bits_per_word: int, number_of_words: int) -> List[int]:
        '''Send command and read multiple SPI data-words, with up to 16 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…16).
            number_of_words (int):
                The number of data-words to read.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''

        buffer_type = typespec_ctypes.c_unsigned_short * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiCmdRead16(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def cmdRead32(self, command_bits: int, command_value: int, dummy_bits: int,
                  transfer_type: int, bits_per_word: int, number_of_words: int) -> List[int]:
        '''Send command and read multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            number_of_words (int):
                The number of data-words to read.

        Returns:
            List[int]: The data-words received.

        Raises:
            DwfLibraryError: An error occurred while executing the read operation.
        '''

        buffer_type = typespec_ctypes.c_unsigned_int * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalSpiCmdRead32(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            rx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        rx_list = list(rx_buffer)

        return rx_list

    def cmdWrite(self, command_bits: int, command_value: int, dummy_bits: int,
                 transfer_type: int, bits_per_word: int, tx: List[int]) -> None:
        '''Send command and write multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            tx (List[int]):
                The data-words to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalSpiCmdWrite(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def cmdWriteOne(self, command_bits: int, command_value: int, dummy_bits: int,
                    transfer_type: int, bits_per_word: int, tx: int) -> None:
        '''Send command and write a single SPI data-word, with up to 32 bits.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits of the data-word (1…32).
            tx (int):
                The data-word to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        result = self.lib.FDwfDigitalSpiCmdWriteOne(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def cmdWrite16(self, command_bits: int, command_value: int, dummy_bits: int,
                   transfer_type: int, bits_per_word: int, tx: List[int]) -> None:
        '''Send command and write multiple SPI data-words, with up to 16 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…16).
            tx (List[int]):
                The data-words to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_short * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalSpiCmdWrite16(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx_buffer,
            number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def cmdWrite32(self, command_bits: int, command_value: int, dummy_bits: int,
                   transfer_type: int, bits_per_word: int, tx: List[int]) -> None:
        '''Send command and write multiple SPI data-words, with up to 32 bits per data-word.

        Parameters:
            command_bits (int):
                The number of command bits.
            command_value (int):
                The command value.
            dummy_bits (int):
                The number of dummy bits before the data transfer.
            transfer_type (int):
                * 0 — SISO
                * 1 — MOSI/MISO
                * 2 — dual
                * 4 — quad
            bits_per_word (int):
                The number of bits per data-word (1…32).
            tx (List[int]):
                The data-words to write.

        Raises:
            DwfLibraryError: An error occurred while executing the write operation.
        '''
        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_int * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalSpiCmdWrite32(
            self.hdwf,
            command_bits,
            command_value,
            dummy_bits,
            transfer_type,
            bits_per_word,
            tx_buffer, number_of_words)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
