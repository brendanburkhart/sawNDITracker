import argparse
import ndi_tool


class ToolDefinition:
    def __init__(self, tool_id, markers, pivot):
        self.id = tool_id
        self.markers = markers
        self.pivot = pivot

    def from_json(json):
        def point_to_array(point):
            return np.array([point.x, point.y, point.z])

        assert json.count == len(json.fiducials)
        pivot = point_to_array(json.pivot)
        markers = [point_to_array(f) for f in json.fiducials]

        tool_id = json.id if "id" in json else None

        return ToolDefinition(tool_id, markers, pivot)

    def to_json(self):
        def array_to_point(array):
            return {"x": array[0], "y": array[1], "z": array[2]}

        fiducials = [array_to_point(m) for m in self.markers]

        json = {
            "count": len(self.markers),
            "fiducials": fiducials,
            "pivot": array_to_point(self.pivot),
        }
    
        if self.id is not None:
            json["id"] = self.id

        return json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", metavar="f", type=str, help="Input file")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        tool = ndi_tool.NDIToolDefinition()
        tool.from_rom(f.read())
        std_tool = tool.to_standard()
        print(std_tool.to_json())

