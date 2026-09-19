
"""
Visionary Fusion -> CV geometry exporter for Fusion 360.

Standalone Fusion 360 script.

What this script does:
    1. Gets the active Fusion 360 design.
    2. Finds all component definitions used by occurrences.
    3. Generates ONE mesh per unique component.
    4. Writes each mesh as an STL in the COMPONENT-LOCAL frame.
    5. Reports component-local dimensions.
    6. Includes CAD -> physical workbench registration metadata.
    7. Writes a JSON manifest describing the exported components.

IMPORTANT:
    - This script does NOT generate top-down silhouettes.
    - This script does NOT export one mesh per occurrence.
    - Meshes are component definitions, not positioned assembly instances.
    - Assembly position and rotation remain in AssemblyGuide.py.

The CV pipeline can therefore take:

    component-local mesh
            +
    occurrence position
            +
    occurrence rotation
            +
    camera / ArUco registration

and generate the expected camera-view silhouette at runtime.

Fusion Design API geometry is internally stored in centimetres.
All exported coordinates and dimensions are converted to millimetres.
"""

import adsk.core
import adsk.fusion
import traceback
import json
import math
import os
import re


# ============================================================
# CONSTANTS
# ============================================================

FUSION_CM_TO_MM = 10.0

OUTPUT_FILENAME = "toCV_output.json"

MESH_DIRECTORY_NAME = "meshes"

# Fusion's mesh calculator quality.
#
# High quality is a reasonable starting point for CV because
# the mesh is being used as geometric reference data rather
# than for rendering.
MESH_QUALITY = adsk.fusion.TriangleMeshQualityOptions.HighQualityTriangleMesh


# ============================================================
# CAD -> WORKBENCH REGISTRATION
# ============================================================
#
# These values are intentionally placeholders.
#
# The CV team will calibrate this using the physical ArUco
# board. The transform is a 2D rigid/similarity transform
# from the Fusion root assembly XY plane to the physical
# workbench coordinate system.
#
# This is NOT the camera extrinsic calibration.
# Camera calibration belongs to the CV side.
#

WORKBENCH_REGISTRATION = {
    "status": "calibration_required",

    "method": "2d_rigid_similarity_transform",

    "rotation_deg": 0.0,

    "translation_mm": [
        0.0,
        0.0
    ],

    "scale": 1.0,

    "source_frame": "fusion_root_assembly_xy",

    "target_frame": "physical_workbench_xy",

    "origin_reference": "physical_workbench_origin_or_aruco_0",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_number(value):
    """Remove tiny floating-point noise and round output."""

    if abs(value) < 0.000001:
        return 0.0

    return round(value, 6)


def sanitize_filename(name):
    """
    Convert a Fusion component name into a filesystem-safe
    filename while preserving the original component ID in
    the JSON manifest.
    """

    value = str(name)

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value
    )

    value = value.strip()

    if not value:
        value = "component"

    return value


def point_to_list_mm(point):
    """Convert a Fusion Point3D from cm to [x, y, z] mm."""

    return [
        clean_number(point.x * FUSION_CM_TO_MM),
        clean_number(point.y * FUSION_CM_TO_MM),
        clean_number(point.z * FUSION_CM_TO_MM)
    ]


def vector_to_list(vector):
    """Convert a Fusion Vector3D into a JSON list."""

    return [
        clean_number(vector.x),
        clean_number(vector.y),
        clean_number(vector.z)
    ]


# ============================================================
# COMPONENT BOUNDING BOX
# ============================================================

def get_component_bounding_box(component):
    """
    Calculate the bounding box of all direct B-Rep bodies
    belonging to a component.

    Coordinates are component-local and converted to mm.
    """

    bodies = component.bRepBodies

    if bodies.count == 0:

        return None

    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")

    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")

    found_solid = False

    for body_index in range(
        bodies.count
    ):

        body = bodies.item(
            body_index
        )

        if not body:
            continue

        try:

            if not body.isSolid:
                continue

        except Exception:

            pass

        box = body.boundingBox

        if not box:
            continue

        found_solid = True

        min_x = min(
            min_x,
            box.minPoint.x
        )

        min_y = min(
            min_y,
            box.minPoint.y
        )

        min_z = min(
            min_z,
            box.minPoint.z
        )

        max_x = max(
            max_x,
            box.maxPoint.x
        )

        max_y = max(
            max_y,
            box.maxPoint.y
        )

        max_z = max(
            max_z,
            box.maxPoint.z
        )

    if not found_solid:
        return None

    minimum = [
        clean_number(
            min_x * FUSION_CM_TO_MM
        ),
        clean_number(
            min_y * FUSION_CM_TO_MM
        ),
        clean_number(
            min_z * FUSION_CM_TO_MM
        )
    ]

    maximum = [
        clean_number(
            max_x * FUSION_CM_TO_MM
        ),
        clean_number(
            max_y * FUSION_CM_TO_MM
        ),
        clean_number(
            max_z * FUSION_CM_TO_MM
        )
    ]

    dimensions = [
        clean_number(
            maximum[axis]
            - minimum[axis]
        )
        for axis in range(3)
    ]

    return {
        "min_mm": minimum,
        "max_mm": maximum,
        "dimensions_mm": dimensions
    }


# ============================================================
# TRIANGLE MESH GENERATION
# ============================================================

def calculate_body_mesh(body):
    """
    Generate a triangular mesh for one B-Rep body.

    The resulting mesh is in the body's native component-local
    coordinate system.

    No occurrence transform is applied here.

    Returns:
        TriangleMesh or None
    """

    try:

        mesh_manager = body.meshManager

    except Exception:

        return None

    if not mesh_manager:
        return None

    try:

        calculator = (
            mesh_manager.createMeshCalculator()
        )

    except Exception:

        return None

    if not calculator:
        return None

    try:

        success = calculator.setQuality(
            MESH_QUALITY
        )

        if not success:
            return None

    except Exception:

        return None

    try:

        mesh = calculator.calculate()

    except Exception:

        return None

    return mesh


# ============================================================
# VECTOR MATH
# ============================================================

def subtract_vectors(a, b):
    """Return a - b."""

    return [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2]
    ]


def cross_product(a, b):
    """Return the 3D cross product."""

    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def vector_length(vector):
    """Return vector magnitude."""

    return math.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )


def normalize_vector(vector):
    """Return normalized vector."""

    length = vector_length(
        vector
    )

    if length <= 1e-12:

        return [
            0.0,
            0.0,
            0.0
        ]

    return [
        vector[0] / length,
        vector[1] / length,
        vector[2] / length
    ]


# ============================================================
# STL WRITING
# ============================================================

def write_ascii_stl(
    file_path,
    component_name,
    triangles
):
    """
    Write triangles to an ASCII STL file.

    The triangle coordinates supplied to this function are
    already in component-local millimetres.

    ASCII STL is deliberately used here because it lets the
    script combine multiple B-Rep bodies belonging to one
    component into ONE component mesh file without relying
    on occurrence transforms.

    STL itself does not store units. The manifest explicitly
    declares millimetres.
    """

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        safe_name = (
            str(component_name)
            .replace("\n", " ")
            .replace("\r", " ")
        )

        file.write(
            "solid {}\n".format(
                safe_name
            )
        )

        for triangle in triangles:

            p1 = triangle[0]
            p2 = triangle[1]
            p3 = triangle[2]

            edge_a = subtract_vectors(
                p2,
                p1
            )

            edge_b = subtract_vectors(
                p3,
                p1
            )

            normal = normalize_vector(
                cross_product(
                    edge_a,
                    edge_b
                )
            )

            file.write(
                "  facet normal {:.9g} {:.9g} {:.9g}\n".format(
                    normal[0],
                    normal[1],
                    normal[2]
                )
            )

            file.write(
                "    outer loop\n"
            )

            file.write(
                "      vertex {:.9g} {:.9g} {:.9g}\n".format(
                    p1[0],
                    p1[1],
                    p1[2]
                )
            )

            file.write(
                "      vertex {:.9g} {:.9g} {:.9g}\n".format(
                    p2[0],
                    p2[1],
                    p2[2]
                )
            )

            file.write(
                "      vertex {:.9g} {:.9g} {:.9g}\n".format(
                    p3[0],
                    p3[1],
                    p3[2]
                )
            )

            file.write(
                "    endloop\n"
            )

            file.write(
                "  endfacet\n"
            )

        file.write(
            "endsolid {}\n".format(
                safe_name
            )
        )


def mesh_to_triangles(mesh):
    """
    Convert a Fusion TriangleMesh into a Python list of
    component-local millimetre triangles.

    Fusion mesh coordinates are in centimetres.
    """

    if not mesh:
        return []

    try:

        coordinates = (
            mesh.nodeCoordinatesAsDouble
        )

        indices = (
            mesh.nodeIndices
        )

    except Exception:

        return []

    if not coordinates:
        return []

    if not indices:
        return []

    triangles = []

    # nodeCoordinatesAsDouble is:
    #
    #   [x0, y0, z0, x1, y1, z1, ...]
    #
    # nodeIndices contains the vertex indices for triangles.

    triangle_count = (
        len(indices) // 3
    )

    for triangle_index in range(
        triangle_count
    ):

        i0 = indices[
            triangle_index * 3
        ]

        i1 = indices[
            triangle_index * 3 + 1
        ]

        i2 = indices[
            triangle_index * 3 + 2
        ]

        base0 = i0 * 3
        base1 = i1 * 3
        base2 = i2 * 3

        if (
            base0 + 2 >= len(coordinates)
            or base1 + 2 >= len(coordinates)
            or base2 + 2 >= len(coordinates)
        ):
            continue

        p0 = [
            clean_number(
                coordinates[base0]
                * FUSION_CM_TO_MM
            ),
            clean_number(
                coordinates[base0 + 1]
                * FUSION_CM_TO_MM
            ),
            clean_number(
                coordinates[base0 + 2]
                * FUSION_CM_TO_MM
            )
        ]

        p1 = [
            clean_number(
                coordinates[base1]
                * FUSION_CM_TO_MM
            ),
            clean_number(
                coordinates[base1 + 1]
                * FUSION_CM_TO_MM
            ),
            clean_number(
                coordinates[base1 + 2]
                * FUSION_CM_TO_MM
            )
        ]

        p2 = [
            clean_number(
                coordinates[base2]
                * FUSION_CM_TO_MM
            ),
            clean_number(
                coordinates[base2 + 1]
                * FUSION_CM_TO_MM
            ),
            clean_number(
                coordinates[base2 + 2]
                * FUSION_CM_TO_MM
            )
        ]

        triangles.append([
            p0,
            p1,
            p2
        ])

    return triangles


# ============================================================
# COMPONENT MESH EXPORT
# ============================================================

def export_component_mesh(
    component,
    output_directory
):
    """
    Export every solid B-Rep body in a component into ONE STL.

    The bodies are meshed in their native component-local
    coordinate system.

    No occurrence transform is applied.
    """

    component_name = component.name

    safe_name = sanitize_filename(
        component_name
    )

    file_name = (
        safe_name
        + ".stl"
    )

    file_path = os.path.join(
        output_directory,
        file_name
    )

    all_triangles = []

    bodies = component.bRepBodies

    body_count = 0

    for body_index in range(
        bodies.count
    ):

        body = bodies.item(
            body_index
        )

        if not body:
            continue

        try:

            if not body.isSolid:
                continue

        except Exception:

            pass

        body_count += 1

        mesh = calculate_body_mesh(
            body
        )

        if not mesh:
            continue

        triangles = mesh_to_triangles(
            mesh
        )

        all_triangles.extend(
            triangles
        )

    if not all_triangles:

        return {
            "status": "no_mesh",
            "filename": None,
            "relative_path": None,
            "triangle_count": 0,
            "body_count": body_count,
            "frame": "component_local",
            "units": "mm"
        }

    write_ascii_stl(
        file_path,
        component_name,
        all_triangles
    )

    return {
        "status": "complete",

        "format": "STL_ascii",

        "filename": file_name,

        "relative_path": (
            MESH_DIRECTORY_NAME
            + "/"
            + file_name
        ),

        "triangle_count": len(
            all_triangles
        ),

        "body_count": body_count,

        "frame": "component_local",

        "units": "mm"
    }


# ============================================================
# COMPONENT COLLECTION
# ============================================================

def collect_unique_components(
    root_component
):
    """
    Find every unique component definition used by an
    occurrence in the assembly.

    Returns a dictionary:

        {
            component_name: Component
        }

    The component itself is exported, rather than an occurrence,
    so its mesh remains in the component-local frame.
    """

    components = {}

    def visit_component(
        component
    ):

        component_id = (
            component.name
        )

        if component_id not in components:

            components[
                component_id
            ] = component

        occurrences = (
            component.occurrences
        )

        for i in range(
            occurrences.count
        ):

            occurrence = (
                occurrences.item(i)
            )

            if not occurrence:
                continue

            try:

                child_component = (
                    occurrence.component
                )

                if child_component:

                    visit_component(
                        child_component
                    )

            except Exception:

                continue

    visit_component(
        root_component
    )

    return components


# ============================================================
# COORDINATE SYSTEM PACKAGE
# ============================================================

def coordinate_system_package(
    registration=None
):
    """
    Describe the CAD and physical workbench coordinate
    systems and their registration transform.
    """

    registration = (
        registration
        or WORKBENCH_REGISTRATION
    )

    theta = math.radians(
        registration.get(
            "rotation_deg",
            0.0
        )
    )

    scale = registration.get(
        "scale",
        1.0
    )

    tx, ty = registration.get(
        "translation_mm",
        [0.0, 0.0]
    )

    c = math.cos(
        theta
    )

    s = math.sin(
        theta
    )

    return {
        "cad_frame": {
            "name":
                "fusion_root_assembly",

            "units":
                "mm",

            "origin":
                "Fusion root component origin",

            "x_axis":
                "Fusion root +X",

            "y_axis":
                "Fusion root +Y",

            "z_axis":
                "Fusion root +Z"
        },

        "component_mesh_frame": {
            "name":
                "component_local",

            "units":
                "mm",

            "description":
                "Each component mesh is exported in the component's native local coordinate system before occurrence position/rotation is applied."
        },

        "workbench_frame": {
            "name":
                "physical_workbench_xy",

            "units":
                "mm",

            "origin":
                registration.get(
                    "origin_reference",
                    "physical_workbench_origin_or_aruco_0"
                ),

            "x_axis":
                "physical workbench +X",

            "y_axis":
                "physical workbench +Y"
        },

        "registration": {
            "status":
                registration.get(
                    "status",
                    "calibration_required"
                ),

            "method":
                registration.get(
                    "method",
                    "2d_rigid_similarity_transform"
                ),

            "rotation_deg":
                clean_number(
                    registration.get(
                        "rotation_deg",
                        0.0
                    )
                ),

            "translation_mm": [
                clean_number(tx),
                clean_number(ty)
            ],

            "scale":
                clean_number(
                    scale
                ),

            "matrix_2d": [
                [
                    clean_number(
                        scale * c
                    ),
                    clean_number(
                        -scale * s
                    ),
                    clean_number(tx)
                ],
                [
                    clean_number(
                        scale * s
                    ),
                    clean_number(
                        scale * c
                    ),
                    clean_number(ty)
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },

        "units": {
            "cad_export":
                "mm",

            "component_mesh":
                "mm",

            "workbench":
                "mm",

            "fusion_internal_length":
                "cm"
        }
    }


# ============================================================
# MAIN EXPORT
# ============================================================

def export_to_cv(
    design,
    output_directory
):
    """
    Generate the complete Fusion -> CV export.

    This replaces the old top-down silhouette exporter.
    """

    root_component = (
        design.rootComponent
    )

    components = (
        collect_unique_components(
            root_component
        )
    )

    mesh_directory = os.path.join(
        output_directory,
        MESH_DIRECTORY_NAME
    )

    os.makedirs(
        mesh_directory,
        exist_ok=True
    )

    export = {
        "format":
            "visionary_toCV",

        "version":
            "2.0",

        "source": {
            "design_name":
                design.parentDocument.name,

            "root_component":
                root_component.name
        },

        "coordinate_system":
            coordinate_system_package(),

        "mesh_export": {
            "format":
                "STL",

            "frame":
                "component_local",

            "units":
                "mm",

            "one_mesh_per":
                "unique_component",

            "directory":
                MESH_DIRECTORY_NAME,

            "quality":
                "high"
        },

        "components": []
    }

    print("")
    print("========================================")
    print("Visionary Fusion -> CV export")
    print("========================================")

    print(
        "Design: {}".format(
            design.parentDocument.name
        )
    )

    print(
        "Unique components: {}".format(
            len(components)
        )
    )

    print("")

    # --------------------------------------------------------
    # Export each unique component.
    # --------------------------------------------------------

    sorted_components = sorted(
        components.items(),
        key=lambda item: item[0]
    )

    for index, item in enumerate(
        sorted_components
    ):

        component_id = item[0]
        component = item[1]

        print(
            "[{}/{}] {}".format(
                index + 1,
                len(sorted_components),
                component_id
            )
        )

        bounding_box = (
            get_component_bounding_box(
                component
            )
        )

        mesh_result = (
            export_component_mesh(
                component,
                mesh_directory
            )
        )

        component_data = {
            "id":
                component_id,

            "mesh":
                mesh_result,

            "bounding_box":
                bounding_box
        }

        export[
            "components"
        ].append(
            component_data
        )

        print(
            "    mesh status: {}".format(
                mesh_result.get(
                    "status",
                    "unknown"
                )
            )
        )

        print(
            "    triangles: {}".format(
                mesh_result.get(
                    "triangle_count",
                    0
                )
            )
        )

        if bounding_box:

            print(
                "    dimensions mm: {}".format(
                    bounding_box[
                        "dimensions_mm"
                    ]
                )
            )

        else:

            print(
                "    dimensions: unavailable"
            )

    # --------------------------------------------------------
    # Write JSON manifest.
    # --------------------------------------------------------

    output_path = os.path.join(
        output_directory,
        OUTPUT_FILENAME
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            export,
            file,
            indent=2
        )

    print("")
    print("========================================")
    print("Export complete")
    print("========================================")

    print(
        "JSON: {}".format(
            output_path
        )
    )

    print(
        "Meshes: {}".format(
            mesh_directory
        )
    )

    print("")

    return export, output_path


# ============================================================
# FUSION 360 ENTRY POINT
# ============================================================

def run(context):
    """
    Fusion 360 calls this function when the script runs.
    """

    ui = None

    try:

        app = (
            adsk.core.Application.get()
        )

        ui = (
            app.userInterface
        )

        # ----------------------------------------------------
        # Active product
        # ----------------------------------------------------

        active_product = (
            app.activeProduct
        )

        if not active_product:

            ui.messageBox(
                "No active Fusion 360 document."
            )

            return

        design = adsk.fusion.Design.cast(
            active_product
        )

        if not design:

            ui.messageBox(
                "The active document is not a Fusion Design."
            )

            return

        # ----------------------------------------------------
        # Output location
        # ----------------------------------------------------

        script_directory = (
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            )
        )

        output_directory = (
            script_directory
        )

        # ----------------------------------------------------
        # Export
        # ----------------------------------------------------

        export, output_path = (
            export_to_cv(
                design,
                output_directory
            )
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        component_count = len(
            export.get(
                "components",
                []
            )
        )

        successful_meshes = sum(
            1
            for component in export.get(
                "components",
                []
            )
            if component.get(
                "mesh",
                {}
            ).get(
                "status"
            ) == "complete"
        )

        ui.messageBox(
            "Visionary Fusion -> CV export complete.\n\n"
            + "Components: "
            + str(component_count)
            + "\n"
            + "Meshes exported: "
            + str(successful_meshes)
            + "\n\n"
            + "JSON:\n"
            + output_path
            + "\n\n"
            + "Mesh directory:\n"
            + os.path.join(
                output_directory,
                MESH_DIRECTORY_NAME
            )
            + "\n\n"
            + "Registration status: "
            + str(
                WORKBENCH_REGISTRATION.get(
                    "status",
                    "unknown"
                )
            )
        )

    except Exception:

        error_message = (
            traceback.format_exc()
        )

        print(
            error_message
        )

        if ui:

            ui.messageBox(
                "toCV.py failed:\n\n"
                + error_message
            )
