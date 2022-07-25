"""
Author: Brendan Burkhart
Created on: 2022-7-22

(C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
"""

import datetime

from struct_definitions import (
    Array,
    Constant,
    Enum,
    Field,
    Float32,
    Padding,
    String,
    Struct,
    UInt8,
    UInt16,
    Vector3f,
    passthrough_property,
)

"""
NDI .rom file format:

All multi-byte fields are little endian
Total file length is 752 bytes, seems to be fixed-length format

Byte range   Description                Field type        Comments
---------------------------------------------------------------------------------------------------
0-2          "NDI"                      ASCII literal
...
4-5          checksum                   uint16            Sum of all later bytes
...
8            unknown                    unknown           Constant 1?
...
12           tool sub type              uint8(?)          Enum, Default 2
...
15           tool main type             uint8             Enum, Default 0
16-17        tool revision              uint16            Default 0
...
20           sequence number            uint10            Default 1, also lower two bits of byte 21
21-23        timestamp                  ---               See SequenceAndDate for details
24           minimum marker angle       uint8             Degrees, [0, 180], default 90
...
28           marker count               uint8             [3, 20]
...
32           minimum markers            uint8             [3, 20], <= marker count
...
36-39        minimum marker error       float32           [0.0, 10.0]
...
72-311       marker geometry            xyz float32       Max of 20
312-551      marker normals             xyz float32       Max of 20
552-554      unknown                    unknown           Constant 0, 1, 2?
...
572-575      unknown                    unknown           Constant 31?
576          unknown                    unknown           Constant 9?
...
580-591      tool manufacturer          ASCII literal     Max length 12 characters
592-593      part number                uint16
...
612          unknown                    unknown           Constant 9?
613-632      marker faces               uint8             Index of face assigned to each marker
633-652      unknown, more faces?       unknown
653-654      unknown                    unknown           Constant 128, 0?
655          marker type                uint8             Enum, Default 41
656-751      face normals               xyz float32       Max 8 faces
"""


class SequenceAndDate(Struct):
    epoch_year = 1900

    sequence_lower = Field("B")
    days = Field("B")
    date_data = Field("B")
    even_years = Field("B")

    def post_decode(self):
        # Days are counted in 64-day blocks, rollovers are counted in lower three bits
        # of next field. middle 4 bits count months, upper bit is set when in odd year
        # Day count incremented by 4 per day - lower two bits used for sequence number
        days = ((self.date_data % 8) << 6) + (self.days >> 2)

        # Years counted by twos starting from 1900
        year_parity = self.date_data >> 7
        years = 2 * self.even_years + self.epoch_year + year_parity

        year_start = datetime.date(years, 1, 1)
        self.date = year_start + datetime.timedelta(days=days)

        months = (self.date_data % 128) >> 3
        if self.date.month != months + 1:
            print("Confusing timestamp! Days since year start doesn't match months!")

        sequence_upper = self.days % 4
        self.sequence_number = self.sequence_lower + 256 * sequence_upper

    def pre_encode(self):
        year = self.date.year
        months = self.date.month - 1 # Zero-indexed
        year_start = datetime.date(year, 1, 1)
        days = (self.date - year_start).days

        year_parity = year % 2

        self.days = (days % 64) << 2 + (self.sequence_number >> 8) % 4
        self.even_years = (year - self.epoch_year) // 2
        self.date_data = (year_parity << 7) + (months << 3) + (days >> 6)
        self.sequence_lower = self.sequence_number % 256


tool_main_types = [
    (0, "Unknown"),
    (1, "Reference"),
    (2, "Pointer"),
    (3, "Button Box"),
    (4, "User Defined"),
    (5, "Microscope"),
    (7, "Calibration Block"),
    (8, "Tool Docking Station"),
    (9, "Isolation Box"),
    (10, "C-Arm Tracker"),
    (11, "Catheter"),
    (12, "GPIO Device"),
    (14, "Scan Reference"),
]

tool_sub_types = [(0, "Removable Tip"), (1, "Fixed Tip"), (2, "Undefined")]

marker_types = {(41, "Passive Sphere"), (49, "Passive Disc"), (57, "Radix Lens")}


class ROMHeader(Struct):
    ndi = Field(String(3))
    p1 = Field(Padding(1))
    checksum = Field(UInt16)
    p2 = Field(Constant([0, 0, 1, 0, 0, 0]))
    tool_sub_type = Field(Enum(tool_sub_types, 2))
    p3 = Field(Padding(2))
    tool_main_type = Field(Enum(tool_main_types, 0))
    tool_revision = Field(UInt16)
    p4 = Field(Padding(2))

    # Sequence and date share part of a byte, need to be parsed together
    sequence_and_date = Field(SequenceAndDate)
    sequence_number = passthrough_property(["sequence_and_date", "sequence_number"])
    date = passthrough_property(["sequence_and_date", "date"])


class ROMGeometry(Struct):
    minimum_marker_angle = Field(UInt8)
    p1 = Field(Padding(3))
    marker_count = Field(UInt8)
    p2 = Field(Padding(3))
    minimum_marker_count = Field(UInt8)
    p3 = Field(Padding(3))
    minimum_marker_error = Field(Float32)
    p4 = Field(Padding(32))
    markers = Field(Array(Vector3f, 20))
    marker_normals = Field(Array(Vector3f, 20))
    p5 = Field(Constant([0, 1, 2]))
    p6 = Field(Padding(17))
    p7 = Field(Constant([31, 31, 31, 31, 9, 0, 0, 0]))


class ROMToolDetails(Struct):
    tool_manufacturer = Field(String(12))
    part_number = Field(UInt16)
    p1 = Field(Padding(18))
    p2 = Field(Constant([9]))


class ROMFaceGeometry(Struct):
    marker_faces = Field(Array(UInt8, 20))
    other_assignments = Field(Array(UInt8, 20))
    p1 = Field(Constant([128, 0]))
    marker_type = Field(Enum(marker_types, 41))
    face_normals = Field(Array(Vector3f, 8))


class NDIROM(Struct):
    header = Field(ROMHeader)
    geometry = Field(ROMGeometry)
    tool_details = Field(ROMToolDetails)
    face_geometry = Field(ROMFaceGeometry)

    # Use post-processing hook to overwrite placeholder checksum
    def post_encode(self, data: bytearray) -> bytearray:
        offset, size = self.header.locate("checksum")
        checksum = sum(data[offset + size :])
        data[offset : offset + size] = self.header.update("checksum", checksum)

        return data
