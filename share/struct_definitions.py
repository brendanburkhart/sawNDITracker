"""
Tools for creating binary format parsers/writers from simple definitions
See ndi_tool.py for examples

Author: Brendan Burkhart
Created on: 2022-7-22

(C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
"""

import inspect
import math
import struct
from typing import Any, List, Tuple, Type

import numpy as np
import numpy.typing as npt


class FieldType:
    """FieldType specifies the type of struct fields, and does (de)serialization"""

    def size(self):
        raise NotImplementedError()

    def default(self):
        raise NotImplementedError()

    def decode(self, data: bytearray):
        raise NotImplementedError()

    def encode(self, value) -> bytearray:
        raise NotImplementedError()


class ByteStruct(FieldType):
    """ByteStruct is a FieldType specified as a format string for the `struct` library"""

    def __init__(self, format: str):
        self.format = struct.Struct(format)

    def size(self):
        return self.format.size

    def default(self):
        return 0

    def decode(self, data: bytearray):
        return self.format.unpack(data)[0]

    def encode(self, value):
        return self.format.pack(value)


def make_field_type(format: FieldType | Type[FieldType] | str) -> FieldType:
    if isinstance(format, FieldType):
        return format
    elif inspect.isclass(format) and issubclass(format, FieldType):
        return format()
    elif isinstance(format, str):
        return ByteStruct(format)
    else:
        raise TypeError(
            "Field parser type must be a FieldType instance, "
            "a FieldType subclass, or a struct format string"
        )


class Field:
    """
    Field is a single field in a Struct, with a specified field type.

    Field instances *should not* be re-used in different Structs or multiple times in the
    same Struct as this can break Field ordering
    """

    # static counter to order Fields by creation order
    id = 0

    def __init__(self, field_type: FieldType | Type[FieldType] | str):
        self.type = make_field_type(field_type)
        self._size = self.type.size()

        self.id = Field.id
        Field.id += 1

    def size(self):
        return self._size

    def decode(self, data: bytearray) -> Tuple[Any, bytearray]:
        if len(data) < self.size:
            raise ValueError("Not enough bytes to complete parsing!")

        remaining_data = data[self.size :]

        value = self.type.decode(data[0 : self.size])
        return value, remaining_data

    def encode(self, value: Any, data: bytearray) -> bytearray:
        encoding = self.type.encode(value)
        assert len(encoding) == self.size
        data.extend(encoding)

        return data


class Vector3f(FieldType):
    """XYZ vector of single-precision 32-bit floats"""

    format = struct.Struct("<3f")

    def __init__(self):
        super().__init__()

    def size(self):
        return Vector3f.format.size

    def default(self):
        return np.array([0.0, 0.0, 0.0])

    def decode(self, data: bytearray) -> npt.ArrayLike:
        values = Vector3f.format.unpack(data)
        return np.array([*values])

    def encode(self, value: npt.ArrayLike) -> bytearray:
        data = bytearray([])

        for v in value:
            data.extend(format.pack(v))

        return data


UInt8 = "<B"
UInt16 = "<H"
Float32 = "<f"
Padding = lambda length: "<{}B".format(length)  # Fixed-size padding


class String(FieldType):
    """
    Fixed-size ASCII string

    Do not use with variable-width formats such as UTF-8
    """

    def __init__(self, length):
        super().__init__()
        self.length = length

    def size(self):
        return self.length

    def default(self):
        return "".join(["\0" for i in range(self.length)])

    def decode(self, data: bytearray) -> str:
        return data.decode("ascii")

    def encode(self, value: str) -> bytearray:
        return bytearray(value.encode("ascii"))


class Array(FieldType):
    """Fixed-length array of specified element type"""

    def __init__(self, element_type, length: int):
        super().__init__()
        self.element_type = make_field_type(element_type)
        self.length = length

    def size(self):
        return self.length * self.element_type.size()

    def default(self):
        default_element = self.element_type.default()
        return np.array([default_element for i in range(self.length)])

    def decode(self, data: bytearray) -> npt.ArrayLike:
        step = self.element_type.size()
        elements = []

        for i in range(self.length):
            element_data = data[i * step : (i + 1) * step]
            elements.append(self.element_type.parse(element_data))

        return np.array(elements)

    def encode(self, value: npt.ArrayLike) -> bytearray:
        data = bytearray([])

        for v in value:
            data.extend(self.element_type.encode(v))

        return data


class Enum(FieldType):
    """
    Enum takes a list of options as tuples of integer value and string name

    Values must be unique and non-negative, however they don't need to be continuous
    or in order
    """

    def __init__(self, options: List[Tuple[int, str]], default: int):
        super().__init__()
        self.options = options

        # Validate options
        distinct_values = set([value for value, name in options])
        if len(distinct_values) != len(options):
            raise ValueError("Invalid Enum options, values must be distinct!")

        for value, name in options:
            if value < 0:
                raise ValueError(
                    "Invalid Enum option ({}, {}), values can't be negative!".format(
                        value, name
                    )
                )

        default_option = [(v, name) for v, name in self.options if v == default]
        if len(default_option) != 1:
            raise ValueError("default Enum value must be one of the provided options")

        self.default_option = default_option[0]

    def default(self):
        return self.default_option

    def size(self):
        highest_option_value = max([value for value, name in self.options])
        bytes_need = math.ceil(math.log(highest_option_value, 2**8))
        return bytes_need

    def decode(self, data: bytearray) -> Tuple[int, str]:
        value = int.from_bytes(data, byteorder="little")
        option = [(v, name) for v, name in self.options if v == value]

        if len(option) != 1:
            raise ValueError("Invalid enum value: {}".format(value))

        return option[0]

    def encode(self, value: Tuple[int, str]) -> bytearray:
        return value[0].to_bytes(length=self.size(), byteorder="little")


class MetaStruct(type):
    """
    Metaclass for struct definitions

    Struct definitions should derive from Struct, not this class. Classes with this
    metaclass can define attributes of type Field, which can be used to
    serialize/deserialize them from byte arrays.
    """

    @staticmethod
    def _make_field_property(key):
        def get(self):
            return self._field_data[key].value

        def set(self, value):
            self._field_data[key].value = value

        return property(get, set)

    @staticmethod
    def _make_decode():
        def decode(self, data: bytearray):
            # Parse fields in order of definition
            keys = sorted(self._fields, key=lambda k: self._fields[k].id)
            struct_value = self.default()

            for key in keys:
                value, data = self._fields[key].decode(data)
                struct_value._field_data[key] = value

            # Post-processing
            struct_value.post_decode()

            return struct_value

        return decode

    @staticmethod
    def _make_encode():
        def encode(self) -> bytearray:
            # Parse fields in order of definition
            keys = sorted(self._fields, key=lambda k: self._fields[k].id)
            data = bytearray([])

            # Pre-precessing hook
            self.pre_encode()

            for key in keys:
                value = self._fields_data[key] or self._fields[key].default()
                data = self._fields[key].encode(value, data)

            # Post-processing hook
            data = self.post_encode(data)
            return data

        return encode

    @staticmethod
    def _make_default():
        def default(self):
            return self.__class__()

        return default

    @staticmethod
    def _make_size():
        def size(self):
            return sum([field.size() for key, field in self._fields.items()])

        return size

    @staticmethod
    def _make_locate():
        def locate(self, key: str):
            # Parse fields in order of definition
            all_keys = sorted(self._fields, key=lambda k: self._fields[k].id)
            preceeding_keys = all_keys[0 : all_keys.index(key)]
            offset = sum([self._fields[key].size() for key in preceeding_keys])
            return offset, self._fields[key].size()

        return locate

    @staticmethod
    def _make_update():
        def update(self, key: str, value):
            self._fields_data[key] = value
            return self._fields[key].encode(value, bytearray([]))

        return update

    def __new__(metaclass, name, bases, attrs):
        bases = (*bases, FieldType)
        new_attrs = {}

        # Store all Fields
        fields = {}

        for key, value in attrs.items():
            if isinstance(value, Field):
                # Transform Field attributes into a @property
                fields[key] = value
                new_attrs[key] = metaclass._make_field_property(key)
            else:
                new_attrs[key] = value

        new_attrs["_fields"] = fields
        new_attrs["decode"] = metaclass._make_decode()
        new_attrs["encode"] = metaclass._make_encode()
        new_attrs["size"] = metaclass._make_size()
        new_attrs["default"] = metaclass._make_default()
        new_attrs["locate"] = metaclass._make_locate()
        new_attrs["update"] = metaclass._make_update()

        return super().__new__(metaclass, name, bases, new_attrs)


class Struct(metaclass=MetaStruct):
    """
    Struct definitions should derive from this base class. Class (not instance)
    attributes of type Field will be used to produce methods to encode/decode
    the Struct to/from bytearray.

    Derived classes must be default-constructible.
    """

    # Initialize instance storage, fill with None
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance._field_data = {key: None for key, value in cls._fields.items()}

        return instance

    # Hook to perform pre-encode processing
    def pre_encode(self):
        pass

    # Hook to perform post-encode processing
    def post_encode(self, data: bytearray) -> bytearray:
        return data

    # Hook to perform post-decode processing
    def post_decode(self):
        pass


def passthrough_property(keys: List[str]):
    def get(self):
        obj = self
        for key in keys:
            obj = getattr(obj, key)

        return obj

    def set(self, value):
        obj = self
        for key in keys[0:-1]:
            obj = getattr(obj, key)

        setattr(obj, keys[-1], value)

    return property(get, set)
