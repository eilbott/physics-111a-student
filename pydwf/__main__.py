#! /usr/bin/env -S python3 -B

'''This is the pydwf support tool that can be executed as 'python -m pydwf'.'''

import zipfile
import io
import os
import re
import base64
import argparse
import importlib

import pydwf
from pydwf import DwfLibrary, DwfEnumConfigInfo


def show_version():
    '''Show the pydwf and DWF library version number.'''

    dwf = DwfLibrary()
    print(f'pydwf version ............ : {pydwf.__version__}')
    print(f'DWF library version ...... : {dwf.getVersion()}')


def list_devices(use_obsolete_api: bool, list_configurations: bool, full_tooltips_flag: bool):
    '''List the devices supported by the DWF library.'''

    # pylint: disable = too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks

    dwf = DwfLibrary()

    num_devices = dwf.deviceEnum.enumerateDevices()

    if num_devices == 0:
        print('No Digilent Waveforms devices found.')

    for device_index in range(num_devices):

        device_type = dwf.deviceEnum.deviceType(device_index)
        is_open = dwf.deviceEnum.deviceIsOpened(device_index)
        user_name = dwf.deviceEnum.userName(device_index)
        device_name = dwf.deviceEnum.deviceName(device_index)
        serial = dwf.deviceEnum.serialNumber(device_index)

        if num_devices == 1:
            header = f'Device information for device #{device_index} ({num_devices} device found)'
        else:
            header = f'Device information for device #{device_index} ({device_index+1} of {num_devices} devices found)'

        print(header)
        print('=' * len(header))
        print()
        print(f'  device ........... : {device_type[0]}')
        print(f'  version .......... : {device_type[1]}')
        print(f'  open ............. : {is_open}')
        print(f'  user_name ........ : {user_name!r}')
        print(f'  device_name ...... : {device_name!r}')
        print(f'  serial ........... : {serial!r}')
        print()

        if use_obsolete_api:

            ai_channels = dwf.deviceEnum.analogInChannels(device_index)
            ai_buffer_size = dwf.deviceEnum.analogInBufferSize(device_index)
            ai_bits = dwf.deviceEnum.analogInBits(device_index)
            ai_frequency = dwf.deviceEnum.analogInFrequency(device_index)

            print('  Analog-in information (obsolete API)')
            print('  ------------------------------------')
            print()
            print(f'  number of channels ...... : {ai_channels!r}')
            print(f'  buffer size ............. : {ai_buffer_size!r}')
            print(f'  bits .................... : {ai_bits!r}')
            print(f'  frequency ............... : {ai_frequency!r}')
            print()

        if list_configurations:

            # This regexp defines the strings that are printed directly (not via the string lookup table).
            re_short_strings = re.compile('^[A-Za-z0-9._]{1,8}$')

            configuration_data: dict[tuple[int, DwfEnumConfigInfo], str] = {}

            num_config = dwf.deviceEnum.enumerateConfigurations(device_index)

            string_lookup_table: dict[str, str] = {}

            for configuration_parameter in DwfEnumConfigInfo:

                for configuration_index in range(num_config):
                    config_value = str(dwf.deviceEnum.configInfo(configuration_index, configuration_parameter))

                    if not full_tooltips_flag and configuration_parameter == DwfEnumConfigInfo.TooltipText:
                        shorthand = f'({len(config_value)})'
                        configuration_data[(configuration_index, configuration_parameter)] = shorthand
                    elif config_value in string_lookup_table:
                        configuration_data[(configuration_index, configuration_parameter)] = \
                            string_lookup_table[config_value]
                    elif not re_short_strings.match(config_value):
                        shorthand = f'[{len(string_lookup_table)}]'
                        string_lookup_table[config_value] = shorthand
                        configuration_data[(configuration_index, configuration_parameter)] = shorthand
                    else:
                        configuration_data[(configuration_index, configuration_parameter)] = config_value

            print(
                '  Configuration:          '
                f'{"  ".join(f"{configuration_index:8d}" for configuration_index in range(num_config))}'
            )
            print(
                '  ----------------------  '
                f'{"  ".join("--------" for configuration_index in range(num_config))}'
            )

            for configuration_parameter in DwfEnumConfigInfo:

                if not full_tooltips_flag and configuration_parameter == DwfEnumConfigInfo.TooltipText:
                    parameter_name = f'{configuration_parameter.name} (length)'
                else:
                    parameter_name = configuration_parameter.name

                print(
                    f'  {parameter_name:22}  '
                    f'{"  ".join(
                        f"{configuration_data[(configuration_index, configuration_parameter)]:>8s}"
                        for configuration_index in range(num_config)
                    )}'
                )
            print()

            if string_lookup_table:

                max_width = 80

                print('  Strings referenced in the preceding table (with newlines replaced by "•"):')
                print()
                for (k, v) in string_lookup_table.items():
                    k = k.replace('\n', '•')
                    idx = 0
                    while True:
                        if idx == 0:
                            print(f'  {v:<4} {k[idx:idx+max_width]!r}')
                        else:
                            print(f'       {k[idx:idx+max_width]!r}')
                        idx += max_width
                        if idx >= len(k):
                            break
                    print()


def extract_zipfile(import_name, target):
    '''Extract zipped, base64-encoded data.'''
    if os.path.exists(target):
        print()
        print(f'Unable to unpack "{target}", the destination path already exists.')
        print()
    else:
        print()
        print(f'Unpacking "{target}" ...')
        print()

        module_name = f'pydwf.data.{import_name}'

        # We only import 'pydwf.data.<target>' if we actually need to access data from it.
        # They can be big (especially the one with the HTML data); importing may take a noticeable amount of time.
        module = importlib.import_module(module_name)

        # Decode and unpack the directory.
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(module.data))) as zip_file:
            zip_file.extractall()


def main():
    '''Parse arguments and execute sub-command.'''

    parser = argparse.ArgumentParser(
        prog='python -m pydwf',
        description=f'Utilities for the pydwf package (version {pydwf.__version__}).',
    )
    subparsers = parser.add_subparsers()

    # If no command is given, execute the top-level parser's 'print_help' method.
    parser.set_defaults(execute=lambda args: parser.print_help())

    # Declare the sub-parser for the 'version' command.
    subparser_version = subparsers.add_parser(
        'version',
        description='Show version of pydwf and the DWF library.',
        help='show version of the DWF library')
    subparser_version.set_defaults(execute=lambda args: show_version())

    # Declare the sub-parser for the 'list' command.

    subparser_list = subparsers.add_parser(
        'list',
        aliases=['ls'],
        description='List Digilent Waveforms devices.',
        help='list Digilent Waveforms devices')

    subparser_list.add_argument(
        '-u', '--use-obsolete-api',
        action='store_true',
        help='for each device, print analog-in parameters obtained using obsolete FDwfEnumAnalogIn* API calls',
        dest='use_obsolete_api')

    subparser_list.add_argument(
        '-c', '--list-configurations',
        action='store_true',
        help='for each device, printing its configurations', dest='list_configurations')

    subparser_list.add_argument(
        '-f', '--full-tooltips',
        action='store_true',
        help='show full TooltipText strings instead of length', dest='full_tooltips')

    subparser_list.set_defaults(
        execute=lambda args: list_devices(args.use_obsolete_api, args.list_configurations, args.full_tooltips))

    # Declare the sub-parser for the 'extract-examples' command.

    subparser_extract_examples = subparsers.add_parser(
        'extract-examples',
        description='Extract pydwf example scripts to "pydwf-examples" directory.',
        help='extract pydwf example scripts to "pydwf-examples" directory')

    subparser_extract_examples.set_defaults(
        execute=lambda args:  extract_zipfile('examples', 'pydwf-examples'))

    # Declare the sub-parser for the 'extract-html-docs' command.

    subparser_extract_html = subparsers.add_parser(
        'extract-html-docs',
        description='Extract pydwf HTML documentation to "pydwf-html-docs" directory.',
        help='extract pydwf HTML documentation to "pydwf-html-docs" directory')

    subparser_extract_html.set_defaults(
        execute=lambda args: extract_zipfile('html_docs', 'pydwf-html-docs'))

    # Declare the sub-parser for the 'extract-html-docs' command.

    subparser_extract_pdf = subparsers.add_parser(
        'extract-pdf-manual',
        description='Extract pydwf PDF manual in current directory.',
        help='extract pydwf PDF manual in current directory')

    subparser_extract_pdf.set_defaults(
        execute=lambda args: extract_zipfile('pdf_manual', f'pydwf-{pydwf.__version__}.pdf'))

    # Parse command-line arguments.
    args = parser.parse_args()

    # Execute the selected command.
    args.execute(args)


if __name__ == '__main__':
    main()
