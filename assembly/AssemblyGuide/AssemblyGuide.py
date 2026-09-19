import adsk.core
import adsk.fusion
import json
import os



def matrix_to_rotation(transform):
    """Extract the 3x3 rotation matrix from a Fusion transform."""

    return [
        [
            transform.getCell(0, 0),
            transform.getCell(0, 1),
            transform.getCell(0, 2)
        ],
        [
            transform.getCell(1, 0),
            transform.getCell(1, 1),
            transform.getCell(1, 2)
        ],
        [
            transform.getCell(2, 0),
            transform.getCell(2, 1),
            transform.getCell(2, 2)
        ]
    ]


def matrix_to_position(transform):
    """Extract XYZ position from a Fusion transform."""

    return [
        transform.getCell(0, 3),
        transform.getCell(1, 3),
        transform.getCell(2, 3)
    ]


def point_to_list(point):
    """Convert Fusion Point3D to [x, y, z]."""

    return [
        point.x,
        point.y,
        point.z
    ]


def get_bounding_box(occurrence):
    """Get the occurrence's bounding box."""

    box = occurrence.boundingBox

    return {
        "min": point_to_list(box.minPoint),
        "max": point_to_list(box.maxPoint)
    }

def get_component_bounds(component):
    """Get the component's geometry bounds in component-local coordinates."""

    bodies = component.bRepBodies

    if bodies.count == 0:
        return None

    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")

    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")

    for body in bodies:

        box = body.boundingBox

        min_x = min(min_x, box.minPoint.x)
        min_y = min(min_y, box.minPoint.y)
        min_z = min(min_z, box.minPoint.z)

        max_x = max(max_x, box.maxPoint.x)
        max_y = max(max_y, box.maxPoint.y)
        max_z = max(max_z, box.maxPoint.z)

    return {
        "min": [
            min_x,
            min_y,
            min_z
        ],
        "max": [
            max_x,
            max_y,
            max_z
        ]
    }

def run(context):

    app = adsk.core.Application.get()
    ui = app.userInterface

    try:

        # --------------------------------------------------
        # Get active Fusion design
        # --------------------------------------------------

        design = adsk.fusion.Design.cast(app.activeProduct)

        if not design:
            ui.messageBox("No Fusion design is currently active.")
            return

        root = design.rootComponent

        # --------------------------------------------------
        # Create assembly structure
        # --------------------------------------------------

        assembly = {
            "assembly": {
                "name": root.name,
                "units": "mm",
                "assembly_axis": [0, 0, 1]
            },
            "components": [],
            "parts": []
        }

        # --------------------------------------------------
        # Go through every occurrence in the root assembly
        # --------------------------------------------------

        components_seen = set()

        for occurrence in root.occurrences:

            transform = occurrence.transform2

            position = matrix_to_position(transform)
            rotation = matrix_to_rotation(transform)

            component_id = occurrence.component.name

            # Add component if we haven't seen it before
            if component_id not in components_seen:
                component = occurrence.component

                assembly["components"].append({
                    "id": component_id,
                    "bounding_box": get_component_bounds(component)
                })

                components_seen.add(component_id)

            # Add this occurrence

            bounding_box = get_bounding_box(occurrence)

            part = {
                "id": occurrence.name,
                "component": component_id,
                "position": position,
                "rotation": rotation,
                "bounding_box": get_bounding_box(occurrence),
                "assembly_height": bounding_box["min"][2]
            }

            assembly["parts"].append(part)

        # --------------------------------------------------
        # Convert to JSON
        # --------------------------------------------------

        json_text = json.dumps(
            assembly,
            indent=4
        )

        # --------------------------------------------------
        # Save JSON to Desktop
        # --------------------------------------------------

        desktop = os.path.expanduser("~/projects/Fusion API/AssemblyGuide")
        file_path = os.path.join(desktop, "assembly.json")

        with open(file_path, "w") as f:
            f.write(json_text)

        # --------------------------------------------------
        # Confirm export
        # --------------------------------------------------

        ui.messageBox(
            "Export complete!\n\nSaved to:\n" + file_path
        )

    except Exception as e:

        ui.messageBox(
            "ERROR:\n\n" + str(e)
        )