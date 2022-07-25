#!/usr/bin/env python

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

import argparse
import json
import pathlib

import numpy as np

import ndi_tool


class SAWToolDefinition:
    def __init__(self, tool_id, markers, pivot=None):
        self.id = tool_id
        self.markers = markers
        self.pivot = pivot

    @staticmethod
    def from_json(json_dict):
        def point_to_array(point):
            return np.array([point["x"], point["y"], point["z"]])

        assert json_dict.get("count", 0) == len(json_dict["fiducials"])
        pivot = point_to_array(json_dict["pivot"]) if "pivot" in json_dict else None
        markers = [point_to_array(f) for f in json_dict["fiducials"]]

        tool_id = json_dict.get("id", None)

        return SAWToolDefinition(tool_id, markers, pivot)

    def to_json(self):
        def array_to_point(array):
            return {"x": array[0], "y": array[1], "z": array[2]}

        json_dict = {}

        if self.id is not None:
            json_dict["id"] = int(self.id)
 
        json_dict["count"] = len(self.markers)
        json_dict["fiducials"] = [array_to_point(m) for m in self.markers]

        if self.pivot is not None:
            json_dict["pivot"] = self.pivot

        return json_dict


def read_rom(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
        tool = ndi_tool.NDIROM.decode(data)

    return tool.to_saw()

def write_rom(tool, file_name):
    tool = ndi_tool.NDIROM.from_saw(tool.id, tool.markers)
    
    with open(file_name, "wb") as f:
        data = ndi_tool.NDIROM.encode(tool)
        f.write(data)

def read_saw(file_name):
    with open(file_name, "r") as f:
        json_dict = json.load(f)
        return SAWToolDefinition.from_json(json_dict)

def write_saw(tool, file_name):
    json_dict = tool.to_json()
    with open(file_name, "w") as f:
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file")
    args = parser.parse_args()

    input_extension = pathlib.Path(args.input).suffix
    output_extension = pathlib.Path(args.output).suffix

    if input_extension == ".rom":
        tool = read_rom(args.input)
    elif input_extension == ".json":
        tool = read_saw(args.input)
    elif input_extension == ".ini":
        raise NotImplemented()
    else:
        raise ValueError("Only NDI .rom, Atracsys .ini, and SAW .json formats are supported!")

    if output_extension == ".rom":
        write_rom(tool, args.output)
    elif output_extension == ".json":
        write_saw(tool, args.output)
    elif output_extension == ".ini":
        raise NotImplemented()
    else:
        raise ValueError("Only NDI .rom, Atracsys .ini, and SAW .json formats are supported!")

