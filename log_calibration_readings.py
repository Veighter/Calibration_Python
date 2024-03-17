import csv
import time
import serial
import numpy as np

# GLOBAL VARIABLES
SER_PORT = 'COM4'  # Serial port
SER_BAUD = 115200  # Serial baud rate
SAMPLE_FREQ = 200  # Frequency to record magnetometer readings at [Hz]
T_SAMPLE = 600 # Total time to read mangetometer readings [sec]
OUTPUT_FILENAME = 'IMU_0.txt'  # Output data file name
SAMPLES=SAMPLE_FREQ*T_SAMPLE



class SerialPort:
    """Create and read data from a serial port.

    Attributes:
        read(**kwargs): Read and decode data string from serial port.
    """
    def __init__(self, port, baud=115200):
        """Create and read serial data.

        Args:
            port (str): Serial port name. Example: 'COM4'
            baud (int): Serial baud rate, default 9600.  """
        if isinstance(port, str) == False:
            raise TypeError('port must be a string.')

        if isinstance(baud, int) == False:
            raise TypeError('Baud rate must be an integer.')

        self.port = port
        self.baud = baud

        # Initialize serial connection
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        self.ser.flushInput()
        self.ser.flushOutput()
    

    def Read(self, clean_end=True):
        """
        Read and decode data string from serial port.

        Args:
            clean_end (bool): Strip '\\r' and '\\n' characters from string. Common if used Serial.println() Arduino function. Default true
        
        Returns:
            (str): utf-8 decoded message.
        """
        # while self.ser.in_waiting:
        bytesToRead = self.ser.readline()
        
        decodedMsg = bytesToRead.decode('utf-8')

        if clean_end == True:
            decodedMsg = decodedMsg.strip('\r').strip('\n')  # Strip extra chars at the end
        
        return decodedMsg
    

    def Write(self, msg):
        """
        Write string to serial port.

        Args
        ----
            msg (str): Message to transmit
        
        Returns
        -------
            (bool) True if successful, false if not
        """
        try:
            self.ser.write(msg.encode())
            return True
        except:
            print("Error sending message.")
    

    def Close(self):
        """Close serial connection."""
        self.ser.close()


teensy = SerialPort(SER_PORT, SER_BAUD)


measurements =[] 


for i in range(SAMPLES):
    data = teensy.Read().split(',')  # Split into separate values
    # "Time [ms], ACC_X [mg], ACC_Y [mg], ACC_Z [mg], GYRO_X [dps], GYRO_Y [dps], GYRO_Z [dps], MAG_X [uT], MAG_Y [uT], MAG_Z [uT]"
    measurements.append([float(data[0]),float(data[1]), float(data[2]), float(data[3]),float(data[4]),float(data[5]), float(data[6]),float(data[7]),float(data[8]),float(data[9])]) 

measurements=np.array(measurements)

# After measurements are complete, write data to file
teensy.Close()
print('Sensor Reading Complete!')

for i in range(SAMPLES):
    with open(OUTPUT_FILENAME, 'a', newline='\n') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([measurements[i, 0], measurements[i, 1], measurements[i, 2],measurements[i,3],measurements[i,4],measurements[i,5],measurements[i,6],measurements[i,7],measurements[i,8],measurements[i,9]])
