"""Acknowledged USB serial client for pan_tilt_servos.ino (115200 baud)."""
import re
import time


class ServoLink:
    def __init__(self, port, timeout=3., serial_factory=None):
        if serial_factory is None:
            import serial
            serial_factory = serial.Serial
        self.serial = serial_factory(port, 115200, timeout=.1, write_timeout=1.)
        self.timeout = timeout

    def __enter__(self):
        try:
            # Opening USB can reset the Uno. STATUS is harmless and retries
            # until a complete response arrives after boot. No HOME on connect.
            deadline = time.monotonic() + 8.
            while time.monotonic() < deadline:
                try:
                    self.positions = self.status()
                    return self
                except TimeoutError:
                    pass
            raise RuntimeError('Arduino did not answer STATUS; upload pan_tilt_servos.ino and close Serial Monitor.')
        except BaseException:
            self.serial.close()
            raise

    def _send(self, command):
        payload = (command + '\n').encode('ascii')
        if self.serial.write(payload) != len(payload):
            raise RuntimeError('Incomplete Arduino serial write.')

    def _lines(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self.serial.readline().decode('ascii', errors='replace').strip()
            if line.startswith(('Rejected.', 'ERROR:', 'Still moving.')):
                raise RuntimeError('Arduino: ' + line)
            if line:
                yield line
        raise TimeoutError('Arduino response timed out; tracking stopped.')

    def status(self):
        self.serial.reset_input_buffer()
        self._send('STATUS')
        positions = {}
        for line in self._lines():
            match = re.fullmatch(r'(Pan|Tilt): commanded (\d+) us; (OFF|HOLDING)', line)
            if match:
                positions[match[1]] = int(match[2])
                if len(positions) == 2:
                    return positions['Pan'], positions['Tilt']

    def move(self, axis, pulse):
        limits = {'P': (400,2700), 'T': (700,1500)}
        if axis not in limits or not isinstance(pulse, int) or not limits[axis][0] <= pulse <= limits[axis][1]:
            raise ValueError('Invalid servo command.')
        self._send(f'{axis} {pulse}')
        expected = f'{"Pan" if axis == "P" else "Tilt"} -> {pulse} us.'
        for line in self._lines():
            if line == expected:
                return

    def off(self):
        self._send('OFF')
        for line in self._lines():
            if line == 'Signals OFF. Support the assembly.':
                return

    def __exit__(self, *_):
        # Keep the last commanded holding position; OFF can drop a loaded head.
        self.serial.close()
