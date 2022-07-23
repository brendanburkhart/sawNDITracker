import argparse
import ndi_tool
import pathlib


class ToolDefinition:
    def __init__(self, tool_id, markers, pivot):
        self.id = tool_id
        self.markers = markers
        self.pivot = pivot

    def from_json(json):
        def point_to_array(point):
            return np.array([point["x"], point["y"], point["z"]])

        assert json.count == len(json["fiducials"])
        pivot = point_to_array(json["pivot"]) if "pivot" in json else None
        markers = [point_to_array(f) for f in json["fiducials"]]

        tool_id = json.get("id", None)

        return ToolDefinition(tool_id, markers, pivot)

    def to_json(self):
        def array_to_point(array):
            return {"x": array[0], "y": array[1], "z": array[2]}

        fiducials = [array_to_point(m) for m in self.markers]

        json = {
            "count": len(self.markers),
            "fiducials": fiducials,
        }
    
        if self.id is not None:
            json["id"] = self.id

        if self.pivot is not None:
            json["pivot"] = self.pivot

        return json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file")
    args = parser.parse_args()

    input_extension = pathlib.Path(args.input).suffix
    output_extension = pathlib.Path(args.output).suffix

    with open(args.input, "rb") as f:
        if input_extension == ".rom":
            tool = ndi_tool.NDIToolDefinition()
            data = f.read()
            tool.from_rom(data)

        std_tool = tool.to_standard()

    with open(args.output, "wb") as f:
        data = tool.to_rom()
        f.write(data)

    print(std_tool.to_json())

