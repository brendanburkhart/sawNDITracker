import datetime
import numpy as np
import struct

import tool_converter

"""
NDI .rom file format:

little endian

byte 0-2:   "NDI"
byte 3: 0?
byte 4-5: checksum - literally sum of bytes 6 to end
byte 8:
byte 12: tool sub type
byte 15: tool main type
byte 16-17: tool revision
byte 20: sequence number, also lower two bits of 21
byte 21-23: timestamp - see parse_timestamp()
byte 28:     marker count
byte 32: minimum marker count?
byte 39: 64 - why?
from byte 72:
    4 byte floats, every 3 is a marker

from byte 312:
    4 byte floats, every 3 is a vectex normal

byte 572-575: 31?
byte 576: 9

---- is position fixed? or depend on number of markers?
bytes 580-?: Manufacturer - find offset from data
bytes 592-593: Part number

byte 612:
byte 613-: face assignments of markers

byte 653-655: 128, 0, 41

byte 656: vec3f face normals, max 8 faces
"""

class NDIToolDefinition:
    marker_data_start = 72
    checksum_bytes = slice(4, 6)

    tool_main_type_byte = 15
    tool_main_types = {
        0: "Unknown",
        1: "Reference",
        2: "Pointer",
        3: "Button Box",
        4: "User Defined",
        5: "Microscope",
        7: "Calibration Block",
        8: "Tool Docking Station",
        9: "Isolation Box",
        10: "C-Arm Tracker",
        11: "Catheter",
        12: "GPIO Device",
        14: "Scan Reference",
    }

    tool_sub_type_byte = 12
    tool_sub_types = {
        0: "Removable Tip",
        1: "Fixed Tip",
        2: "Undefined",
    }

    marker_type_byte = 655
    marker_types = {
        41: "Passive Sphere",
        49: "Passive Disc",
        57: "Radix Lens",
    }

    tool_revision_bytes = slice(16,18)
    sequence_number_byte = 20
    part_number_bytes = slice(592, 594)

    epoch_year = 1900
    timestamp_start_byte = 21

    @staticmethod
    def from_standard(tool_id, markers, pivot):
        self.part_number = tool_id
        self.markers = markers

    def to_standard(self):
        return tool_converter.ToolDefinition(self.part_number, self.markers, self.pivot)
    
    def _parse_timestamp(self, rom: bytes):
        # Years counted by twos starting from 1900
        years = 2 * rom[self.timestamp_start_byte+2] + self.epoch_year

        # Days are counted in 64-day blocks, rollovers are counted in
        # lower three bits of next field. Next 4 bits count months,
        # Next bit is set when in odd year
        data = rom[self.timestamp_start_byte+1]
        day_rollovers = data % 8
        year_rollovers = data // 128
        months = (data % 128) // 8
        # Day count incremented by 4 per day - lower two bits used for sequence number
        days = rom[self.timestamp_start_byte] // 4

        years += year_rollovers
        # 16 days per rollover
        days += 64 * day_rollovers

        timestamp = datetime.date(years, 1, 1)
        timestamp = timestamp + datetime.timedelta(days=days)

        if timestamp.month != months+1:
            print("Confusing timestamp! Days since year start doesn't match months!")

        return timestamp

    def _compute_checksum(self, data: bytes):
        running_sum = 0
        for byte in data:
            running_sum += int(byte)
    
        return running_sum

    def from_rom(self, rom: bytes):
        if len(rom) <= self.marker_data_start:
            print("Not a valid NDI .rom file: file header is too short")
            return

        if rom[0:3].decode("utf-8") != "NDI":
            print(str(rom[0:3]))
            print("Not a valid NDI .rom file: doesn't start with 'NDI'")
            return

        tool_main_type = rom[self.tool_main_type_byte]
        if tool_main_type not in self.tool_main_types:
            print("Unknown tool main type: {}".format(int(tool_main_type)))
        else:
            self.tool_main_type = self.tool_main_types[tool_main_type]
            print("Main type: {}".format(self.tool_main_type))

        tool_sub_type = rom[self.tool_sub_type_byte]
        if tool_sub_type not in self.tool_sub_types:
            print("Unknown tool sub type: {}".format(int(tool_sub_type)))
        else:
            self.tool_sub_type = self.tool_sub_types[tool_sub_type]
            print("Sub type: {}".format(self.tool_sub_type))

        marker_type = rom[self.marker_type_byte]
        if marker_type not in self.marker_types:
            print("Unknown marker type: {}".format(int(marker_type)))
        else:
            self.marker_type = self.marker_types[marker_type]
            print("Marker type: {}".format(self.marker_type))

        self.tool_revision = int.from_bytes(rom[self.tool_revision_bytes], byteorder='little')
        print("Tool revision: {}".format(self.tool_revision))
    
        self.sequence_number = int(rom[self.sequence_number_byte]) + 256*(rom[self.sequence_number_byte+1] % 4)
        print("Sequence number: {}".format(self.sequence_number))
 
        self.part_number = int.from_bytes(rom[self.part_number_bytes], byteorder='little')
        print("Part number: {}".format(self.part_number))

        print(self._parse_timestamp(rom))

        computed_checksum = self._compute_checksum(rom[6:])
        checksum = int.from_bytes(rom[self.checksum_bytes], byteorder='little')
        if computed_checksum != checksum:
            print("Incorrect checksum! Should be {:d} but is {:d}".format(computed_checksum, checksum))

        marker_count = int.from_bytes(rom[28:29], byteorder="little")
        self.markers = []

        # struct format: little endian, 3 floats
        marker_format = "<3f"
        stride = struct.calcsize(marker_format)

        for i in range(marker_count):
            x, y, z = struct.unpack(marker_format, rom[i*stride:(i+1)*stride])
            self.markers.append(np.array([x, y, z], dtype=np.float32))

        self.pivot = np.array([0.0, 0.0, 0.0])

