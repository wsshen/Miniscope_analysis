"""
File:                       digital_in.py

Library Call Demonstrated:  mcculw.ul.d_in() and mcculw.ul.d_bit_in()

Purpose:                    Reads a digital input port.

Demonstration:              Configures the first compatible port  for input
                            (if necessary) and then reads and displays the value
                            on the port and the first bit.

Other Library Calls:        mcculw.ul.d_config_port()
                            mcculw.ul.release_daq_device()

Special Requirements:       Device must have a digital input port
                            or have digital ports programmable as input.
"""
from __future__ import absolute_import, division, print_function
from builtins import *  # @UnusedWildImport

from mcculw import ul
from mcculw.enums import DigitalIODirection
from mcculw.enums import InterfaceType
from mcculw.device_info import DaqDeviceInfo
import timeit
import keyboard

def run_example():
    # By default, the example detects and displays all available devices and
    # selects the first device listed. Use the dev_id_list variable to filter
    # detected devices by device ID (see UL documentation for device IDs).
    # If use_device_detection is set to False, the board_num variable needs to
    # match the desired board number configured with Instacal.
    use_device_detection = True
    dev_id_list = []
    board_num = 0

    try:
        ul.ignore_instacal()
        if use_device_detection:
            devices = ul.get_daq_device_inventory(InterfaceType.USB)
            device = devices[0]
            ul.create_daq_device(board_num, device)

        daq_dev_info = DaqDeviceInfo(board_num)
        if not daq_dev_info.supports_digital_io:
            raise Exception('Error: The DAQ device does not support '
                            'digital I/O')

        print('\nActive DAQ device: ', daq_dev_info.product_name, ' (',
              daq_dev_info.unique_id, ')\n', sep='')

        dio_info = daq_dev_info.get_dio_info()

        # Find the first port that supports input, defaulting to None
        # if one is not found.
        port = next((port for port in dio_info.port_info if port.supports_input),
                    None)
        if not port:
            raise Exception('Error: The DAQ device does not support '
                            'digital input')

        # If the port is configurable, configure it for input.
        if port.is_port_configurable:
            ul.d_config_port(board_num, port.type, DigitalIODirection.IN)

        # Get a value from the digital port
        bit_num = 2
        last_port_value = 0
        last_start_time = 0
        stim_points = ""
        while True:
            if keyboard.is_pressed('q'):
                break
            port_value = ul.d_in(board_num, port.type)
            if port_value != last_port_value:
                start_time = timeit.default_timer()
                print("Port value changed. Port value is:", port_value, " Time is:", start_time)
                stim_points += str(start_time)+'\t'+str(port_value)+'\t'
                last_port_value = port_value
        with open("stims.txt", "w", newline="") as f:
            f.write(stim_points)

    except Exception as e:
        print('\n', e)
    finally:
        if use_device_detection:
            ul.release_daq_device(board_num)


if __name__ == '__main__':
    run_example()
