#!/usr/bin/env python

'''
Sketch to control Ardunio for transmission LED ring and fluorescence LED trigger
'''

import time
import serial

# serial_port_var is the handle of the arduino serial port, e.g.   serial_port_var = serial.Serial(port='COM3', baudrate=115200, timeout=.1)= serial.Serial(port='COM3', baudrate=115200, timeout=2)
# arduino = Serial(port=arduinoCom, baudrate=115200, timeout=.1)
# x= 'a' sets up LED ring pattern 1 to turn on during camera TTL output. Function returns 'a' to confirm that it has been turned on
# x= 'b' sets up LED ring pattern 2 to turn on during camera TTL output. Function returns 'b' to confirm that it has been turned on
# x= 'c' sets up LED ring pattern 3 to turn on during camera TTL output. Function returns 'c' to confirm that it has been turned on
# x= 'd' sets up LED ring pattern 4 to turn on during camera TTL output. Function returns 'd' to confirm that it has been turned on
# x= 'l' sets up red LED to turn on during camera TTL output. Function returns 'l' to confirm that it has been turned on
# x= 'f' blocks camera TTL to triggerscope. Function returns 'f' to confirm that it has been turned off


def init_arduino(port,baudrate,timeout):

    arduino_port = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    return arduino_port

def set_state(arduino_port, x):
    arduino_port.flushInput
    arduino_port.flushOutput
    arduino_port.write(bytes(x, 'utf-8'))
    time.sleep(0.01)
    data = arduino_port.readline().decode('ascii').strip('\r\n')
    return data