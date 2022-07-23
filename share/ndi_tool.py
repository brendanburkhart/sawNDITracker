import datetime
import numpy as np
import struct

import tool_converter

"""
NDI .rom file format:

little endian

byte 0-2:     "NDI"
byte 3:       0?
byte 4-5:     checksum - literally sum of bytes 6 to end
byte 8:       1?
byte 12:      tool sub type
byte 15:      tool main type
byte 16-17:   tool revision
byte 20:      sequence number, also lower two bits of 21
byte 21-23:   timestamp - see parse_timestamp()
byte 24:      minimum marker angle, degrees, int8
byte 28:      marker count
byte 32:      minimum marker count
    If tool has many markers, can track with just subset,
    this configures how many the tracker should require
byte 39:      64 - why?
byte 72-311:  xyz point, float32 markers, max of 20
byte 312-551: xyz vector, float32 marker normals
byte 552-554: 0, 1, 2?
byte 572-575: 31?
byte 576:     9
byte 580-?:   Tool manufacturer - find offset from data
byte 592-593: Part number

byte 612:     9?
byte 613-632: face assignments of markers
byte 633-652: more assignments?
byte 653-655: 128, 0, 41
byte 656-751: xyz vector, float32 face normals, max 8 faces

total length: 752 bytes, seems to be fixed
"""

class NDIToolDefinition:
    marker_count_byte = 28
    minimum_markers_byte = 32
    marker_data_start = 72
    marker_normals_bytes = slice(312, 552)
    checksum_bytes = slice(4, 6)

    minimum_marker_angle_byte = 24

    ndi_rom_length = 752

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
    default_main_type = 4

    tool_sub_type_byte = 12
    tool_sub_types = {
        0: "Removable Tip",
        1: "Fixed Tip",
        2: "Undefined",
    }
    default_sub_type = 2

    marker_type_byte = 655
    marker_types = {
        41: "Passive Sphere",
        49: "Passive Disc",
        57: "Radix Lens",
    }
    default_marker_type = 41

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
        return tool_converter.ToolDefinition(self.part_number, self.markers, None)
    
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

    def _parse_vec3f(self, data: bytes):
        # struct format: little endian, 3 floats
        data_format = "<3f"
        stride = struct.calcsize(data_format)

        assert(len(data) % stride == 0)

        count = len(data) // stride
        vectors = []

        for i in range(count):
            x, y, z = struct.unpack(data_format, data[i*stride:(i+1)*stride])
            vectors.append(np.array([x, y, z], dtype=np.float32))

        return vectors

    def _write_vec3f(self, vectors):
        data_format = "<3f"
        data = []
        for v in vectors:
            data.extend(struct.pack(data_format, *v))

        return bytes(data)

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

        marker_count = int(rom[self.marker_count_byte])
        self.markers = self._parse_vec3f(rom[self.marker_data_start:marker_count*12+self.marker_data_start])

    def to_rom(self):
        data = bytearray(self.ndi_rom_length)

        data[0:3] = "NDI".encode("utf-8")
        data[self.marker_type_byte] = self.default_marker_type
        data[self.tool_main_type_byte] = self.default_main_type
        data[self.tool_sub_type_byte] = self.default_sub_type
        data[self.part_number_bytes] = self.part_number.to_bytes(2, byteorder='little')
        data[self.marker_count_byte] = len(self.markers)
        data[self.minimum_markers_byte] = 3
        marker_data = self._write_vec3f(self.markers)
        data[self.marker_data_start:self.marker_data_start+len(marker_data)] = marker_data

        # Unknown data
        data[8] = 1
        data[39] = 64
        data[552] = 0
        data[553] = 1
        data[554] = 2
        data[572:576] = [31, 31, 31, 31]
        data[612] = 9
        data[653] = 128
        data[655] = 41

        # Timestamp
        data[21:24] = [40, 51, 61]

        data[self.minimum_marker_angle_byte] = 90 # Degrees

        # Assign all markers to face 1
        # Assign marker normals
        for i in range(len(self.markers)):
            data[613+i] = 1
            data[633+i] = 1
            data[312+12*i:312+12*(i+1)] = struct.pack("<3f", 0, 0, 1)

        data[656:656+12] = struct.pack("<3f", 0, 0, 1)

        checksum = self._compute_checksum(data[6:])
        checksum_length = len(data[self.checksum_bytes])
        data[self.checksum_bytes] = checksum.to_bytes(checksum_length, byteorder='little')

        return data


