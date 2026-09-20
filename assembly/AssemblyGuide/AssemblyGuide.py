import adsk.core
import adsk.fusion
import json
import os
import math


# ==========================================================
# Configuration
# ==========================================================

# Fusion's API stores design lengths in centimetres.
# AssemblyGuide exports millimetres.
FUSION_CM_TO_MM = 10.0

# Broad-phase geometric contact tolerance.
GEOMETRIC_CONTACT_TOLERANCE_MM = 1.0

# Set this to a root occurrence name to override automatic
# anchor selection.
ASSEMBLY_ANCHOR_OVERRIDE = None

# Optional component-specific reference points in mm.
#
# Example:
#
# COMPONENT_REFERENCE_POINT_OVERRIDES = {
#     "Base Plate": {
#         "name": "bottom_front_left_mounting_corner",
#         "local_position": [0.0, 0.0, 0.0]
#     }
# }
#
COMPONENT_REFERENCE_POINT_OVERRIDES = {}


# ==========================================================
# Utility functions
# ==========================================================

def clean_number(value):
    """Remove tiny floating-point errors and round values."""

    if abs(value) < 0.000001:
        return 0.0

    return round(value, 6)


def clean_vector(vector):
    """Round a 3D vector."""

    return [
        clean_number(vector[0]),
        clean_number(vector[1]),
        clean_number(vector[2])
    ]


def matrix_to_rotation(transform):
    """
    Extract the 3x3 rotation matrix from a Fusion Matrix3D.
    """

    return [
        [
            clean_number(transform.getCell(0, 0)),
            clean_number(transform.getCell(0, 1)),
            clean_number(transform.getCell(0, 2))
        ],
        [
            clean_number(transform.getCell(1, 0)),
            clean_number(transform.getCell(1, 1)),
            clean_number(transform.getCell(1, 2))
        ],
        [
            clean_number(transform.getCell(2, 0)),
            clean_number(transform.getCell(2, 1)),
            clean_number(transform.getCell(2, 2))
        ]
    ]


def matrix_to_position(transform):
    """
    Extract XYZ translation from a Fusion transform.

    Fusion translation is in cm; output is mm.
    """

    return [
        clean_number(transform.getCell(0, 3) * FUSION_CM_TO_MM),
        clean_number(transform.getCell(1, 3) * FUSION_CM_TO_MM),
        clean_number(transform.getCell(2, 3) * FUSION_CM_TO_MM)
    ]


def point_to_list(point):
    """
    Convert a Fusion Point3D from cm to mm.
    """

    return [
        clean_number(point.x * FUSION_CM_TO_MM),
        clean_number(point.y * FUSION_CM_TO_MM),
        clean_number(point.z * FUSION_CM_TO_MM)
    ]


def transform_local_point_to_assembly(
    position,
    rotation,
    local_point
):
    """
    Transform a component-local point into Fusion root assembly
    coordinates.

    p_assembly = R * p_local + t

    All values are mm.
    """

    return [
        clean_number(
            position[row]
            + sum(
                rotation[row][column] * local_point[column]
                for column in range(3)
            )
        )
        for row in range(3)
    ]


def vector_length(vector):
    """Return the magnitude of a 3D vector."""

    return math.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )


def normalize_vector(vector):
    """Return a normalized 3D vector."""

    length = vector_length(vector)

    if length <= 0.000001:
        return [0.0, 0.0, 0.0]

    return [
        vector[0] / length,
        vector[1] / length,
        vector[2] / length
    ]


def dot_product(vector_a, vector_b):
    """Return the dot product of two 3D vectors."""

    return sum(
        vector_a[i] * vector_b[i]
        for i in range(3)
    )


def cross_product(vector_a, vector_b):
    """Return the cross product of two 3D vectors."""

    return [
        vector_a[1] * vector_b[2]
        - vector_a[2] * vector_b[1],

        vector_a[2] * vector_b[0]
        - vector_a[0] * vector_b[2],

        vector_a[0] * vector_b[1]
        - vector_a[1] * vector_b[0]
    ]


def transpose_rotation(rotation):
    """
    Return the transpose/inverse of an orthonormal 3x3 rotation.
    """

    return [
        [
            rotation[column][row]
            for column in range(3)
        ]
        for row in range(3)
    ]


def multiply_rotation_matrices(left, right):
    """Return left * right for two 3x3 matrices."""

    return [
        [
            clean_number(
                sum(
                    left[row][index]
                    * right[index][column]
                    for index in range(3)
                )
            )
            for column in range(3)
        ]
        for row in range(3)
    ]


def multiply_rotation_vector(rotation, vector):
    """Return R * v."""

    return [
        clean_number(
            sum(
                rotation[row][column]
                * vector[column]
                for column in range(3)
            )
        )
        for row in range(3)
    ]


def vectors_are_close(
    vector_a,
    vector_b,
    tolerance=0.000001
):
    """Determine whether two vectors are effectively identical."""

    for i in range(3):

        if abs(
            vector_a[i]
            - vector_b[i]
        ) > tolerance:

            return False

    return True


# ==========================================================
# Coordinate-frame analysis
# ==========================================================

def get_component_axis_in_assembly(
    rotation,
    local_axis
):
    """
    Transform a component-local axis into assembly coordinates.

    local_axis is normally one of:
        [1,0,0]
        [0,1,0]
        [0,0,1]
    """

    return normalize_vector(
        multiply_rotation_vector(
            rotation,
            local_axis
        )
    )


def get_assembly_axis_in_component(
    rotation,
    assembly_axis
):
    """
    Transform an assembly-space axis into component-local
    coordinates.

    Because R is orthonormal:

        R^-1 = R^T

    This is particularly useful for determining which local
    direction corresponds to assembly +Z.

    Example:

        local +Y -> assembly +Z

    produces approximately:

        assembly +Z -> local +Y
    """

    inverse_rotation = transpose_rotation(
        rotation
    )

    return normalize_vector(
        multiply_rotation_vector(
            inverse_rotation,
            assembly_axis
        )
    )


def classify_cardinal_axis(vector):
    """
    If a vector is essentially aligned with X/Y/Z, identify it.

    Otherwise return None.

    This is diagnostic only. Arbitrary orientations remain valid.
    """

    vector = normalize_vector(vector)

    axes = {
        "+X": [1.0, 0.0, 0.0],
        "-X": [-1.0, 0.0, 0.0],
        "+Y": [0.0, 1.0, 0.0],
        "-Y": [0.0, -1.0, 0.0],
        "+Z": [0.0, 0.0, 1.0],
        "-Z": [0.0, 0.0, -1.0]
    }

    best_name = None
    best_dot = -1.0

    for name, axis in axes.items():

        score = dot_product(
            vector,
            axis
        )

        if score > best_dot:

            best_dot = score
            best_name = name

    if best_dot >= 0.999:

        return best_name

    return None


def calculate_in_plane_yaw(
    rotation,
    assembly_axis,
    component_up_axis
):
    """
    Calculate the component's in-plane yaw around the assembly axis.

    IMPORTANT:

    We do not assume that the component's local +Z is the
    component's up direction.

    The component_up_axis is explicitly supplied.

    For the common case:

        component local +Y -> assembly +Z

    the local forward/right basis is constructed from the
    remaining local axes.

    The result is the angle of the component's local +X axis
    projected into the assembly plane.

    Returns degrees.
    """

    assembly_axis = normalize_vector(
        assembly_axis
    )

    component_up_axis = normalize_vector(
        component_up_axis
    )

    # Transform local +X into assembly space.
    local_x_assembly = get_component_axis_in_assembly(
        rotation,
        [1.0, 0.0, 0.0]
    )

    # Project local +X onto the assembly plane.
    projected_x = [
        local_x_assembly[i]
        - dot_product(
            local_x_assembly,
            assembly_axis
        ) * assembly_axis[i]
        for i in range(3)
    ]

    projected_x = normalize_vector(
        projected_x
    )

    if vector_length(projected_x) <= 0.000001:

        return {
            "status": "undefined",
            "reason": "local_x_parallel_to_assembly_axis",
            "degrees": None
        }

    # We need a stable in-plane reference direction.
    #
    # Use global +X unless it is parallel to the assembly axis,
    # then use global +Y.
    if abs(
        dot_product(
            assembly_axis,
            [1.0, 0.0, 0.0]
        )
    ) < 0.95:

        reference_x = [1.0, 0.0, 0.0]

    else:

        reference_x = [0.0, 1.0, 0.0]

    # Project reference direction into assembly plane.
    reference_x = normalize_vector([
        reference_x[i]
        - dot_product(
            reference_x,
            assembly_axis
        ) * assembly_axis[i]
        for i in range(3)
    ])

    reference_y = normalize_vector(
        cross_product(
            assembly_axis,
            reference_x
        )
    )

    x_component = dot_product(
        projected_x,
        reference_x
    )

    y_component = dot_product(
        projected_x,
        reference_y
    )

    yaw = math.degrees(
        math.atan2(
            y_component,
            x_component
        )
    )

    return {
        "status": "valid",
        "degrees": clean_number(yaw),
        "reference_axis": "assembly_plane_reference_x",
        "reference_x": clean_vector(reference_x),
        "reference_y": clean_vector(reference_y),
        "component_local_axis_used": "+X",
        "component_up_axis": clean_vector(
            component_up_axis
        )
    }


# ==========================================================
# Component reference points
# ==========================================================

def get_component_reference_point(
    component_id,
    component_bounds
):
    """
    Return the named local reference point for a component.

    Default:
        component-local bounding-box minimum corner.
    """

    override = (
        COMPONENT_REFERENCE_POINT_OVERRIDES.get(
            component_id
        )
    )

    if override is not None:

        local_position = override.get(
            "local_position"
        )

        if (
            isinstance(local_position, list)
            and len(local_position) == 3
        ):

            return {
                "name": override.get(
                    "name",
                    "configured_component_reference"
                ),

                "method":
                    "configured_component_reference",

                "cad_local_position": [
                    clean_number(value)
                    for value in local_position
                ],

                "local_coordinate_frame":
                    "component_local",

                "units":
                    "mm"
            }

        raise ValueError(
            "Reference-point override for '"
            + component_id
            + "' must contain a three-value local_position."
        )

    if component_bounds is None:

        return {
            "name":
                "component_local_origin",

            "method":
                "component_bounds_unavailable",

            "cad_local_position":
                [0.0, 0.0, 0.0],

            "local_coordinate_frame":
                "component_local",

            "units":
                "mm"
        }

    return {
        "name":
            "component_local_bounding_box_min_corner",

        "method":
            "component_local_bounding_box_minimum",

        "cad_local_position":
            component_bounds["min"],

        "local_coordinate_frame":
            "component_local",

        "units":
            "mm"
    }


# ==========================================================
# Bounding boxes
# ==========================================================

def get_bounding_box(occurrence):
    """
    Get occurrence bounding box in root assembly coordinates.
    """

    box = occurrence.boundingBox

    return {
        "min": point_to_list(
            box.minPoint
        ),

        "max": point_to_list(
            box.maxPoint
        )
    }


def get_component_bounds(component):
    """
    Get component geometry bounds in component-local coordinates.
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

    for body in bodies:

        box = body.boundingBox

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

    return {
        "min": [
            clean_number(
                min_x * FUSION_CM_TO_MM
            ),
            clean_number(
                min_y * FUSION_CM_TO_MM
            ),
            clean_number(
                min_z * FUSION_CM_TO_MM
            )
        ],

        "max": [
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
    }


def get_bbox_dimensions(box):
    """Return [x, y, z] dimensions."""

    return [
        max(
            0.0,
            box["max"][axis]
            - box["min"][axis]
        )
        for axis in range(3)
    ]


def get_axis_minimum_from_bbox(
    box,
    axis_index
):
    """
    Return the minimum coordinate along a global
    cardinal assembly axis.
    """

    return clean_number(
        box["min"][axis_index]
    )


# ==========================================================
# Geometry relationship detection
# ==========================================================

def bounding_boxes_close(
    box_a,
    box_b,
    tolerance=1.0
):
    """
    Determine whether two bounding boxes overlap or are
    within tolerance.

    This is broad-phase contact detection only.
    """

    for axis in range(3):

        a_min = box_a["min"][axis]
        a_max = box_a["max"][axis]

        b_min = box_b["min"][axis]
        b_max = box_b["max"][axis]

        if a_max < b_min:

            gap = b_min - a_max

        elif b_max < a_min:

            gap = a_min - b_max

        else:

            gap = 0.0

        if gap > tolerance:
            return False

    return True


def find_geometric_relationships(
    occurrences,
    tolerance=1.0
):
    """Find broad-phase candidate contact pairs."""

    relationships = []

    for i in range(
        len(occurrences)
    ):

        for j in range(
            i + 1,
            len(occurrences)
        ):

            a = occurrences[i]
            b = occurrences[j]

            if bounding_boxes_close(
                a["bounding_box"],
                b["bounding_box"],
                tolerance
            ):

                relationships.append({
                    "part_a":
                        a["id"],

                    "part_b":
                        b["id"],

                    "type":
                        "candidate_contact",

                    "confidence":
                        "low"
                })

    return relationships


# ==========================================================
# Fusion joint extraction
# ==========================================================

def get_joint_relationships(root):
    """
    Extract Fusion joints associated with occurrences.

    Uses occurrence.joints.

    Duplicate pairs are removed.
    """

    relationships = []
    seen = set()

    for occurrence in root.occurrences:

        try:
            joints = occurrence.joints

        except Exception:
            continue

        for i in range(
            joints.count
        ):

            try:
                joint = joints.item(i)

            except Exception:
                continue

            if not joint:
                continue

            occurrence_one = None
            occurrence_two = None

            try:
                occurrence_one = (
                    joint.occurrenceOne
                )

            except Exception:
                pass

            try:
                occurrence_two = (
                    joint.occurrenceTwo
                )

            except Exception:
                pass

            if (
                not occurrence_one
                or not occurrence_two
            ):
                continue

            id_one = occurrence_one.name
            id_two = occurrence_two.name

            pair_key = tuple(
                sorted([
                    id_one,
                    id_two
                ])
            )

            if pair_key in seen:
                continue

            seen.add(pair_key)

            joint_type = "unknown"

            try:
                joint_type = str(
                    joint.objectType
                )

            except Exception:
                pass

            relationships.append({
                "part_a":
                    id_one,

                "part_b":
                    id_two,

                "type":
                    "joint",

                "joint_type":
                    joint_type,

                "confidence":
                    "high"
            })

    return relationships


# ==========================================================
# Merge relationship evidence
# ==========================================================

def merge_relationships(
    geometric_relationships,
    joint_relationships
):
    """Merge geometric and Fusion-joint evidence."""

    relationship_map = {}

    for relationship in geometric_relationships:

        key = tuple(
            sorted([
                relationship["part_a"],
                relationship["part_b"]
            ])
        )

        relationship_map[key] = {
            "part_a":
                relationship["part_a"],

            "part_b":
                relationship["part_b"],

            "evidence": [
                {
                    "type":
                        "candidate_contact",

                    "confidence":
                        "low"
                }
            ]
        }

    for relationship in joint_relationships:

        key = tuple(
            sorted([
                relationship["part_a"],
                relationship["part_b"]
            ])
        )

        if key not in relationship_map:

            relationship_map[key] = {
                "part_a":
                    relationship["part_a"],

                "part_b":
                    relationship["part_b"],

                "evidence": []
            }

        relationship_map[key][
            "evidence"
        ].append({
            "type":
                "joint",

            "joint_type":
                relationship["joint_type"],

            "confidence":
                "high"
        })

    return list(
        relationship_map.values()
    )


# ==========================================================
# Assembly-axis identification
# ==========================================================

def get_axis_interval_metrics(
    box_a,
    box_b,
    axis_index
):
    """Calculate interface metrics along one global axis."""

    dimensions_a = get_bbox_dimensions(
        box_a
    )

    dimensions_b = get_bbox_dimensions(
        box_b
    )

    overlaps = []
    gaps = []
    overlap_ratios = []

    for axis in range(3):

        a_min = box_a["min"][axis]
        a_max = box_a["max"][axis]

        b_min = box_b["min"][axis]
        b_max = box_b["max"][axis]

        overlap = max(
            0.0,
            min(
                a_max,
                b_max
            )
            - max(
                a_min,
                b_min
            )
        )

        if a_max < b_min:

            gap = b_min - a_max

        elif b_max < a_min:

            gap = a_min - b_max

        else:

            gap = 0.0

        reference_dimension = min(
            dimensions_a[axis],
            dimensions_b[axis]
        )

        if reference_dimension > 0.000001:

            overlap_ratio = min(
                1.0,
                max(
                    0.0,
                    overlap
                    / reference_dimension
                )
            )

        else:

            overlap_ratio = 0.0

        overlaps.append(
            overlap
        )

        gaps.append(
            gap
        )

        overlap_ratios.append(
            overlap_ratio
        )

    axis_overlap_ratio = (
        overlap_ratios[axis_index]
    )

    perpendicular_axes = [
        axis
        for axis in range(3)
        if axis != axis_index
    ]

    perpendicular_overlap = (
        sum(
            overlap_ratios[axis]
            for axis in perpendicular_axes
        )
        / 2.0
    )

    separation_score = (
        1.0
        - axis_overlap_ratio
    )

    interface_score = (
        separation_score
        * perpendicular_overlap
    )

    return {
        "overlap":
            clean_number(
                overlaps[axis_index]
            ),

        "gap":
            clean_number(
                gaps[axis_index]
            ),

        "overlap_ratio":
            clean_number(
                axis_overlap_ratio
            ),

        "separation_score":
            clean_number(
                separation_score
            ),

        "perpendicular_overlap":
            clean_number(
                perpendicular_overlap
            ),

        "interface_score":
            clean_number(
                interface_score
            ),

        "all_axis_overlap_ratios": [
            clean_number(value)
            for value in overlap_ratios
        ]
    }


def get_relationship_weight(
    relationship
):
    """Return confidence weight for relationship evidence."""

    has_joint = False
    has_candidate_contact = False

    for evidence in relationship.get(
        "evidence",
        []
    ):

        if evidence.get(
            "type"
        ) == "joint":

            has_joint = True

        elif evidence.get(
            "type"
        ) == "candidate_contact":

            has_candidate_contact = True

    if has_joint:
        return 2.0

    if has_candidate_contact:
        return 0.5

    return 0.0


def get_relationship_evidence_types(
    relationship
):
    """Return relationship evidence types."""

    return [
        evidence.get("type")
        for evidence in relationship.get(
            "evidence",
            []
        )
    ]


def analyze_global_assembly_axis(
    parts,
    relationships
):
    """
    Identify the primary global assembly axis.

    Interface geometry is the primary signal.

    The axis direction sign is determined separately.
    """

    axis_names = [
        "X",
        "Y",
        "Z"
    ]

    axis_vectors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    part_map = {
        part["id"]: part
        for part in parts
    }

    axis_totals = [
        0.0,
        0.0,
        0.0
    ]

    axis_weights = [
        0.0,
        0.0,
        0.0
    ]

    relationship_diagnostics = []

    for relationship in relationships:

        part_a = part_map.get(
            relationship["part_a"]
        )

        part_b = part_map.get(
            relationship["part_b"]
        )

        if not part_a or not part_b:
            continue

        weight = get_relationship_weight(
            relationship
        )

        if weight <= 0.0:
            continue

        axis_metrics = []

        for axis_index in range(3):

            metrics = (
                get_axis_interval_metrics(
                    part_a["bounding_box"],
                    part_b["bounding_box"],
                    axis_index
                )
            )

            axis_metrics.append(
                metrics
            )

            axis_totals[axis_index] += (
                weight
                * metrics["interface_score"]
            )

            axis_weights[axis_index] += (
                weight
            )

        center_a = part_a["position"]
        center_b = part_b["position"]

        center_difference = [
            center_b[i]
            - center_a[i]
            for i in range(3)
        ]

        relationship_diagnostics.append({
            "part_a":
                part_a["id"],

            "part_b":
                part_b["id"],

            "evidence":
                get_relationship_evidence_types(
                    relationship
                ),

            "weight":
                clean_number(weight),

            "interface_geometry": {
                "X":
                    axis_metrics[0],

                "Y":
                    axis_metrics[1],

                "Z":
                    axis_metrics[2]
            },

            "center_difference_diagnostic":
                clean_vector(
                    center_difference
                )
        })

    axis_scores = []

    for axis_index in range(3):

        if axis_weights[axis_index] > 0.0:

            score = (
                axis_totals[axis_index]
                / axis_weights[axis_index]
            )

        else:

            score = 0.0

        axis_scores.append(
            clean_number(score)
        )

    high_confidence_relationship_count = sum(
        1
        for relationship
        in relationship_diagnostics
        if relationship["weight"] >= 2.0
    )

    if max(axis_weights) <= 0.0:

        selected_index = 2
        confidence = "low"

        selection_reason = (
            "no_relationship_evidence"
        )

    else:

        selected_index = max(
            range(3),
            key=lambda index:
                axis_scores[index]
        )

        sorted_scores = sorted(
            axis_scores,
            reverse=True
        )

        margin = (
            sorted_scores[0]
            - sorted_scores[1]
        )

        if (
            high_confidence_relationship_count >= 2
            and margin >= 0.15
        ):

            confidence = "high"

        elif margin >= 0.05:

            confidence = "medium"

        else:

            confidence = "low"

        selection_reason = (
            "highest_weighted_interface_alignment"
        )

    signed_direction = 0.0

    for relationship in relationship_diagnostics:

        part_a = part_map.get(
            relationship["part_a"]
        )

        part_b = part_map.get(
            relationship["part_b"]
        )

        if not part_a or not part_b:
            continue

        weight = relationship["weight"]

        delta = (
            part_b["position"][selected_index]
            - part_a["position"][selected_index]
        )

        signed_direction += (
            weight * delta
        )

    if signed_direction < 0.0:

        selected_vector = [
            -axis_vectors[selected_index][0],
            -axis_vectors[selected_index][1],
            -axis_vectors[selected_index][2]
        ]

    else:

        selected_vector = axis_vectors[
            selected_index
        ]

    return {
        "method":
            "relationship_interface_geometry",

        "axis_selection_reason":
            selection_reason,

        "axis_scores": {
            "X":
                axis_scores[0],

            "Y":
                axis_scores[1],

            "Z":
                axis_scores[2]
        },

        "axis_weights": {
            "X":
                clean_number(
                    axis_weights[0]
                ),

            "Y":
                clean_number(
                    axis_weights[1]
                ),

            "Z":
                clean_number(
                    axis_weights[2]
                )
        },

        "selected_axis":
            axis_names[selected_index],

        "selected_axis_vector":
            selected_vector,

        "confidence":
            confidence,

        "relationship_count":
            len(
                relationship_diagnostics
            ),

        "high_confidence_relationship_count":
            high_confidence_relationship_count,

        "relationship_analysis":
            relationship_diagnostics
    }


# ==========================================================
# Candidate direction generation
# ==========================================================

def add_direction(
    directions,
    name,
    vector,
    frame,
    source
):
    """Add a unique normalized direction."""

    vector = normalize_vector(
        vector
    )

    if vector_length(vector) <= 0.000001:
        return

    for existing in directions:

        if vectors_are_close(
            existing["vector"],
            vector
        ):

            return

    directions.append({
        "name":
            name,

        "vector":
            clean_vector(vector),

        "frame":
            frame,

        "source":
            source
    })


def generate_candidate_directions(
    rotation
):
    """Generate global and local candidate directions."""

    directions = []

    # Global axes.

    add_direction(
        directions,
        "+X_global",
        [1, 0, 0],
        "global",
        "global_axis"
    )

    add_direction(
        directions,
        "-X_global",
        [-1, 0, 0],
        "global",
        "global_axis"
    )

    add_direction(
        directions,
        "+Y_global",
        [0, 1, 0],
        "global",
        "global_axis"
    )

    add_direction(
        directions,
        "-Y_global",
        [0, -1, 0],
        "global",
        "global_axis"
    )

    add_direction(
        directions,
        "+Z_global",
        [0, 0, 1],
        "global",
        "global_axis"
    )

    add_direction(
        directions,
        "-Z_global",
        [0, 0, -1],
        "global",
        "global_axis"
    )

    # Local axes transformed into assembly coordinates.

    local_x = get_component_axis_in_assembly(
        rotation,
        [1, 0, 0]
    )

    local_y = get_component_axis_in_assembly(
        rotation,
        [0, 1, 0]
    )

    local_z = get_component_axis_in_assembly(
        rotation,
        [0, 0, 1]
    )

    add_direction(
        directions,
        "+x_local",
        local_x,
        "local",
        "local_axis"
    )

    add_direction(
        directions,
        "-x_local",
        [
            -local_x[0],
            -local_x[1],
            -local_x[2]
        ],
        "local",
        "local_axis"
    )

    add_direction(
        directions,
        "+y_local",
        local_y,
        "local",
        "local_axis"
    )

    add_direction(
        directions,
        "-y_local",
        [
            -local_y[0],
            -local_y[1],
            -local_y[2]
        ],
        "local",
        "local_axis"
    )

    add_direction(
        directions,
        "+z_local",
        local_z,
        "local",
        "local_axis"
    )

    add_direction(
        directions,
        "-z_local",
        [
            -local_z[0],
            -local_z[1],
            -local_z[2]
        ],
        "local",
        "local_axis"
    )

    return directions


def analyze_candidate_directions(
    parts
):
    """Generate candidate directions for every part."""

    results = []

    for part in parts:

        directions = (
            generate_candidate_directions(
                part["rotation"]
            )
        )

        results.append({
            "part":
                part["id"],

            "candidate_directions":
                directions
        })

    return results


# ==========================================================
# Joint graph
# ==========================================================

def get_joint_graph(
    relationships
):
    """Build an undirected graph from Fusion joints."""

    graph = {}

    for relationship in relationships:

        has_joint_evidence = False

        for evidence in relationship.get(
            "evidence",
            []
        ):

            if evidence.get(
                "type"
            ) == "joint":

                has_joint_evidence = True
                break

        if not has_joint_evidence:
            continue

        part_a = relationship["part_a"]
        part_b = relationship["part_b"]

        if part_a not in graph:
            graph[part_a] = []

        if part_b not in graph:
            graph[part_b] = []

        if part_b not in graph[part_a]:
            graph[part_a].append(
                part_b
            )

        if part_a not in graph[part_b]:
            graph[part_b].append(
                part_a
            )

    return graph


# ==========================================================
# Assembly heights
# ==========================================================

def get_part_height_map(
    parts,
    assembly_axis
):
    """
    Calculate scalar position along the assembly axis.

    For a cardinal axis, use the minimum occurrence bounding-box
    coordinate along that axis. This preserves the original
    'lowest resting plane' meaning.

    For an arbitrary axis, project the occurrence origin.
    """

    axis = normalize_vector(
        assembly_axis
    )

    cardinal_axis = None

    if vectors_are_close(
        axis,
        [1, 0, 0]
    ):

        cardinal_axis = 0

    elif vectors_are_close(
        axis,
        [0, 1, 0]
    ):

        cardinal_axis = 1

    elif vectors_are_close(
        axis,
        [0, 0, 1]
    ):

        cardinal_axis = 2

    heights = {}

    for part in parts:

        if cardinal_axis is not None:

            height = get_axis_minimum_from_bbox(
                part["bounding_box"],
                cardinal_axis
            )

        else:

            position = part["position"]

            height = dot_product(
                position,
                axis
            )

        heights[
            part["id"]
        ] = clean_number(
            height
        )

    return heights


# ==========================================================
# Precedence
# ==========================================================

def make_precedence_edge(
    before,
    after,
    height_before,
    height_after,
    reason
):
    """Create one directed precedence edge."""

    return {
        "before":
            before,

        "after":
            after,

        "reason":
            reason,

        "evidence":
            "fusion_joint",

        "confidence":
            "high",

        "height_before":
            clean_number(
                height_before
            ),

        "height_after":
            clean_number(
                height_after
            )
    }


def analyze_joint_precedence(
    parts,
    relationships,
    assembly_axis,
    height_tolerance=0.001
):
    """
    Convert Fusion-joint connectivity into a partial order.

    Lower resting part -> higher part.

    Equal-height pairs remain ambiguous.
    """

    heights = get_part_height_map(
        parts,
        assembly_axis
    )

    joint_graph = get_joint_graph(
        relationships
    )

    precedence_edges = []
    ambiguous_pairs = []

    processed_pairs = set()

    for part_a in joint_graph:

        for part_b in joint_graph[part_a]:

            pair_key = tuple(
                sorted([
                    part_a,
                    part_b
                ])
            )

            if pair_key in processed_pairs:
                continue

            processed_pairs.add(
                pair_key
            )

            height_a = heights.get(
                part_a,
                0.0
            )

            height_b = heights.get(
                part_b,
                0.0
            )

            height_difference = (
                height_b
                - height_a
            )

            if abs(
                height_difference
            ) <= height_tolerance:

                ambiguous_pairs.append({
                    "part_a":
                        part_a,

                    "part_b":
                        part_b,

                    "reason":
                        "same_assembly_height",

                    "evidence":
                        "fusion_joint",

                    "confidence":
                        "high"
                })

                continue

            if height_a < height_b:

                precedence_edges.append(
                    make_precedence_edge(
                        part_a,
                        part_b,
                        height_a,
                        height_b,
                        "lower_part_precedes_higher_part"
                    )
                )

            else:

                precedence_edges.append(
                    make_precedence_edge(
                        part_b,
                        part_a,
                        height_b,
                        height_a,
                        "lower_part_precedes_higher_part"
                    )
                )

    precedence_graph = {}

    for part in joint_graph:
        precedence_graph[part] = []

    for edge in precedence_edges:

        before = edge["before"]
        after = edge["after"]

        if before not in precedence_graph:
            precedence_graph[before] = []

        if after not in precedence_graph:
            precedence_graph[after] = []

        if after not in precedence_graph[
            before
        ]:

            precedence_graph[
                before
            ].append(
                after
            )

    incoming_count = {}

    for part in precedence_graph:
        incoming_count[part] = 0

    for before in precedence_graph:

        for after in precedence_graph[
            before
        ]:

            incoming_count[after] += 1

    roots = []
    leaves = []

    for part in precedence_graph:

        if incoming_count[part] == 0:
            roots.append(part)

        if len(
            precedence_graph[part]
        ) == 0:

            leaves.append(part)

    roots.sort()
    leaves.sort()

    # Connected components.

    connected_components = []
    visited = set()

    for start in sorted(
        joint_graph.keys()
    ):

        if start in visited:
            continue

        component = []
        stack = [start]

        visited.add(start)

        while stack:

            current = stack.pop()

            component.append(
                current
            )

            for neighbor in joint_graph.get(
                current,
                []
            ):

                if neighbor not in visited:

                    visited.add(
                        neighbor
                    )

                    stack.append(
                        neighbor
                    )

        component.sort()

        connected_components.append(
            component
        )

    return {
        "method":
            "fusion_joint_graph_plus_assembly_axis_height",

        "assembly_axis":
            assembly_axis,

        "part_heights":
            heights,

        "joint_graph":
            joint_graph,

        "precedence_graph":
            precedence_graph,

        "precedence_edges":
            precedence_edges,

        "ambiguous_pairs":
            ambiguous_pairs,

        "roots":
            roots,

        "leaves":
            leaves,

        "connected_components":
            connected_components
    }


# ==========================================================
# Assembly plan
# ==========================================================

def get_part_reference_position(
    part
):
    """Return the assembly-space reference point."""

    reference_point = part.get(
        "reference_point",
        {}
    )

    return reference_point.get(
        "assembly_position",
        part["position"]
    )


def pose_relative_to_anchor(
    anchor_part,
    target_part
):
    """
    Express target final pose in anchor reference-point coordinates.
    """

    anchor_rotation_inverse = (
        transpose_rotation(
            anchor_part["rotation"]
        )
    )

    anchor_position = (
        get_part_reference_position(
            anchor_part
        )
    )

    target_position = (
        get_part_reference_position(
            target_part
        )
    )

    position_delta = [
        target_position[axis]
        - anchor_position[axis]
        for axis in range(3)
    ]

    return {
        "position":
            multiply_rotation_vector(
                anchor_rotation_inverse,
                position_delta
            ),

        "rotation":
            multiply_rotation_matrices(
                anchor_rotation_inverse,
                target_part["rotation"]
            )
    }


def select_assembly_anchor(
    part_by_id,
    incoming_count,
    precedence_analysis,
    anchor_override=None
):
    """
    Select a root part as the live assembly anchor.
    """

    candidate_ids = sorted([
        part_id
        for part_id, count
        in incoming_count.items()
        if (
            count == 0
            and part_id in part_by_id
        )
    ])

    if not candidate_ids:

        return {
            "status":
                "no_anchor_candidate",

            "part":
                None,

            "candidates":
                []
        }

    part_heights = (
        precedence_analysis.get(
            "part_heights",
            {}
        )
    )

    candidates = [
        {
            "part":
                part_id,

            "assembly_axis_projection_mm":
                clean_number(
                    part_heights.get(
                        part_id,
                        0.0
                    )
                )
        }
        for part_id in candidate_ids
    ]

    if anchor_override is not None:

        if anchor_override not in candidate_ids:

            return {
                "status":
                    "invalid_anchor_override",

                "part":
                    None,

                "requested_part":
                    anchor_override,

                "candidates":
                    candidates
            }

        return {
            "status":
                "configured",

            "method":
                "configured_root_override",

            "part":
                anchor_override,

            "candidates":
                candidates
        }

    selected_part = min(
        candidate_ids,
        key=lambda part_id: (
            part_heights.get(
                part_id,
                0.0
            ),
            part_id
        )
    )

    return {
        "status":
            "automatic",

        "method":
            "root_lowest_along_assembly_axis",

        "part":
            selected_part,

        "candidates":
            candidates,

        "tie_breaker":
            "occurrence_name_ascending"
    }


def generate_assembly_plan(
    parts,
    precedence_analysis,
    anchor_override=None
):
    """
    Generate a staged topological assembly plan.

    Independent operations may occur in the same step.

    The anchor is explicitly placed first so that subsequent
    anchor-relative operations have an established physical frame.
    """

    precedence_graph = (
        precedence_analysis.get(
            "precedence_graph",
            {}
        )
    )

    all_part_ids = [
        part["id"]
        for part in parts
    ]

    graph = {}

    part_by_id = {
        part["id"]:
            part
        for part in parts
    }

    for part_id in all_part_ids:
        graph[part_id] = []

    for before in precedence_graph:

        if before not in graph:
            graph[before] = []

        for after in precedence_graph[
            before
        ]:

            if after not in graph:
                graph[after] = []

            if after not in graph[
                before
            ]:

                graph[
                    before
                ].append(
                    after
                )

    incoming_count = {
        part_id: 0
        for part_id in graph
    }

    for before in graph:

        for after in graph[
            before
        ]:

            incoming_count[after] += 1

    anchor_selection = (
        select_assembly_anchor(
            part_by_id,
            incoming_count,
            precedence_analysis,
            anchor_override
        )
    )

    if anchor_selection["part"] is None:

        return {
            "method":
                "precedence_graph_topological_planning",

            "status":
                "error",

            "error":
                anchor_selection["status"],

            "anchor_selection":
                anchor_selection,

            "steps":
                []
        }

    anchor_part = part_by_id[
        anchor_selection["part"]
    ]

    remaining = set(
        graph.keys()
    )

    completed = set()

    steps = []
    step_number = 1

    # ======================================================
    # Anchor must be physically established first.
    # ======================================================

    anchor_id = anchor_part["id"]

    anchor_dependencies = []

    for dependency in graph:

        if anchor_id in graph[
            dependency
        ]:

            anchor_dependencies.append(
                dependency
            )

    if anchor_dependencies:

        return {
            "method":
                "precedence_graph_topological_planning",

            "status":
                "error",

            "error":
                "anchor_has_unresolved_dependencies",

            "anchor":
                anchor_id,

            "dependencies":
                sorted(
                    anchor_dependencies
                ),

            "steps":
                []
        }

    anchor_reference_point = (
        anchor_part.get(
            "reference_point",
            {}
        )
    )

    steps.append({
        "step":
            step_number,

        "operations": [
            {
                "id":
                    "place_" + anchor_id,

                "type":
                    "PLACE",

                "part":
                    anchor_id,

                "component":
                    anchor_part["component"],

                "target_position":
                    get_part_reference_position(
                        anchor_part
                    ),

                "target_position_reference": {
                    "name":
                        anchor_reference_point.get(
                            "name",
                            "occurrence_origin"
                        ),

                    "method":
                        anchor_reference_point.get(
                            "method",
                            "legacy_occurrence_origin"
                        ),

                    "reference_frame":
                        anchor_reference_point.get(
                            "reference_frame",
                            "occurrence_origin_frame"
                        ),

                    "reference_frame_position":
                        [0.0, 0.0, 0.0]
                },

                "target_rotation":
                    anchor_part["rotation"],

                "target_coordinate_frame":
                    "fusion_root_assembly",

                "target_units":
                    "mm",

                "target_relative_to_anchor": {
                    "anchor_part":
                        anchor_id,

                    "position":
                        [0.0, 0.0, 0.0],

                    "rotation":
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]
                        ],

                    "coordinate_frame":
                        "anchor_reference_point_local",

                    "position_units":
                        "mm"
                },

                "dependencies":
                    [],

                "depends_on_parts":
                    []
            }
        ]
    })

    completed.add(
        anchor_id
    )

    remaining.remove(
        anchor_id
    )

    step_number += 1

    # ======================================================
    # Remaining topological planning.
    # ======================================================

    while remaining:

        available = []

        for part_id in sorted(
            remaining
        ):

            dependencies_satisfied = True

            for dependency in graph:

                if part_id in graph[
                    dependency
                ]:

                    if dependency not in completed:

                        dependencies_satisfied = False
                        break

            if dependencies_satisfied:

                available.append(
                    part_id
                )

        if not available:

            return {
                "method":
                    "precedence_graph_topological_planning",

                "status":
                    "error",

                "error":
                    "precedence_graph_cycle",

                "unresolved_parts":
                    sorted(remaining),

                "steps":
                    steps
            }

        operations = []

        for part_id in available:

            part = part_by_id[
                part_id
            ]

            dependencies = sorted([
                dependency
                for dependency in graph
                if part_id in graph[
                    dependency
                ]
            ])

            relative_pose = (
                pose_relative_to_anchor(
                    anchor_part,
                    part
                )
            )

            reference_point = (
                part.get(
                    "reference_point",
                    {}
                )
            )

            operations.append({
                "id":
                    "place_" + part_id,

                "type":
                    "PLACE",

                "part":
                    part_id,

                "component":
                    part["component"],

                "target_position":
                    get_part_reference_position(
                        part
                    ),

                "target_position_reference": {
                    "name":
                        reference_point.get(
                            "name",
                            "occurrence_origin"
                        ),

                    "method":
                        reference_point.get(
                            "method",
                            "legacy_occurrence_origin"
                        ),

                    "reference_frame":
                        reference_point.get(
                            "reference_frame",
                            "occurrence_origin_frame"
                        ),

                    "reference_frame_position":
                        [0.0, 0.0, 0.0]
                },

                "target_rotation":
                    part["rotation"],

                "target_coordinate_frame":
                    "fusion_root_assembly",

                "target_units":
                    "mm",

                "target_relative_to_anchor": {
                    "anchor_part":
                        anchor_part["id"],

                    "position":
                        relative_pose["position"],

                    "rotation":
                        relative_pose["rotation"],

                    "coordinate_frame":
                        "anchor_reference_point_local",

                    "position_units":
                        "mm"
                },

                "dependencies": [
                    "place_" + dependency
                    for dependency in dependencies
                ],

                "depends_on_parts":
                    dependencies
            })

        steps.append({
            "step":
                step_number,

            "operations":
                operations
        })

        for part_id in available:

            remaining.remove(
                part_id
            )

            completed.add(
                part_id
            )

        step_number += 1

    return {
        "method":
            "precedence_graph_topological_planning",

        "status":
            "valid",

        "coordinate_system": {
            "frame":
                "fusion_root_assembly",

            "units":
                "mm",

            "origin":
                "Fusion root component origin"
        },

        "assembly_anchor": {
            "part":
                anchor_part["id"],

            "operation":
                "place_" + anchor_part["id"],

            "reference_point":
                anchor_part.get(
                    "reference_point",
                    {
                        "name":
                            "occurrence_origin",

                        "assembly_position":
                            anchor_part["position"],

                        "units":
                            "mm"
                    }
                ),

            "selection":
                anchor_selection
        },

        "steps":
            steps,

        "operation_count":
            sum(
                len(
                    step["operations"]
                )
                for step in steps
            ),

        "step_count":
            len(steps)
    }


# ==========================================================
# Main
# ==========================================================

def run(context):

    app = adsk.core.Application.get()
    ui = app.userInterface

    try:

        # ==================================================
        # Active design
        # ==================================================

        design = adsk.fusion.Design.cast(
            app.activeProduct
        )

        if not design:

            ui.messageBox(
                "No Fusion design is currently active."
            )

            return

        root = design.rootComponent

        # ==================================================
        # Base export structure
        # ==================================================

        assembly = {
            "assembly": {
                "name":
                    root.name,

                "units":
                    "mm",

                "coordinate_frame":
                    "fusion_root_assembly"
            },

            "coordinate_system": {
                "cad_root_frame":
                    "fusion_root_assembly",

                "component_frame":
                    "component_local",

                "workbench_frame":
                    "workbench",

                "camera_frame":
                    "camera",

                "registration": {
                    "status":
                        "calibration_required",

                    "method":
                        "cad_anchor_to_aruco_then_camera_extrinsics",

                    "cad_to_aruco": {
                        "translation_mm":
                            None,

                        "rotation_deg":
                            None
                    },

                    "note":
                        "Physical anchor registration must be calibrated on the workbench."
                }
            },

            "camera_model": {
                "status":
                    "calibration_required",

                "projection":
                    "perspective",

                "note":
                    "Actual camera projection is performed in CV using calibrated intrinsics and extrinsics."
            },

            "components": [],

            "parts": [],

            "relationships": [],

            "direction_analysis": [],

            "precedence_analysis": {},

            "assembly_plan": {}
        }

        # ==================================================
        # Component tracking
        # ==================================================

        components_seen = set()

        component_bounds_by_id = {}

        component_reference_by_id = {}

        # ==================================================
        # Occurrence information
        # ==================================================

        occurrences = []

        # ==================================================
        # Export every root occurrence
        # ==================================================

        for occurrence in root.occurrences:

            transform = occurrence.transform2

            position = matrix_to_position(
                transform
            )

            rotation = matrix_to_rotation(
                transform
            )

            bounding_box = get_bounding_box(
                occurrence
            )

            component_name = (
                occurrence.component.name
            )

            # ----------------------------------------------
            # Stable component ID
            # ----------------------------------------------

            component_id = component_name

            if component_id in components_seen:

                suffix = 2

                while (
                    component_id
                    + "__"
                    + str(suffix)
                ) in components_seen:

                    suffix += 1

                component_id = (
                    component_name
                    + "__"
                    + str(suffix)
                )

            component = occurrence.component

            # ----------------------------------------------
            # Component definition
            # ----------------------------------------------

            if component_id not in components_seen:

                component_bounds = (
                    get_component_bounds(
                        component
                    )
                )

                component_reference_point = (
                    get_component_reference_point(
                        component_id,
                        component_bounds
                    )
                )

                # If a duplicate component ID was generated,
                # fall back to the actual Fusion component name
                # for overrides.
                if (
                    component_reference_point[
                        "method"
                    ]
                    == "component_local_bounding_box_minimum"
                    and component_name
                    != component_id
                ):

                    component_reference_point = (
                        get_component_reference_point(
                            component_name,
                            component_bounds
                        )
                    )

                component_bounds_by_id[
                    component_id
                ] = component_bounds

                component_reference_by_id[
                    component_id
                ] = component_reference_point

                assembly["components"].append({
                    "id":
                        component_id,

                    "fusion_component_name":
                        component_name,

                    "bounding_box":
                        component_bounds,

                    "reference_point": {
                        "name":
                            component_reference_point[
                                "name"
                            ],

                        "method":
                            component_reference_point[
                                "method"
                            ],

                        "reference_frame":
                            "component_reference_point_local",

                        "reference_frame_position":
                            [0.0, 0.0, 0.0],

                        "units":
                            "mm",

                        "cad_mapping": {
                            "source_frame":
                                "component_local",

                            "position":
                                component_reference_point[
                                    "cad_local_position"
                                ]
                        }
                    }
                })

                components_seen.add(
                    component_id
                )

            component_reference_point = (
                component_reference_by_id[
                    component_id
                ]
            )

            # ----------------------------------------------
            # Reference point in assembly coordinates
            # ----------------------------------------------

            reference_position = (
                transform_local_point_to_assembly(
                    position,
                    rotation,
                    component_reference_point[
                        "cad_local_position"
                    ]
                )
            )

            # ----------------------------------------------
            # Part's assembly-axis height is filled after
            # axis detection below.
            # ----------------------------------------------

            # ----------------------------------------------
            # Part
            # ----------------------------------------------

            part = {
                "id":
                    occurrence.name,

                "component":
                    component_id,

                "fusion_component_name":
                    component_name,

                "position":
                    position,

                "occurrence_origin":
                    position,

                "rotation":
                    rotation,

                "reference_point": {
                    "name":
                        component_reference_point[
                            "name"
                        ],

                    "method":
                        component_reference_point[
                            "method"
                        ],

                    "reference_frame":
                        "part_reference_point_local",

                    "reference_frame_position":
                        [0.0, 0.0, 0.0],

                    "assembly_position":
                        reference_position,

                    "coordinate_frame":
                        "fusion_root_assembly",

                    "units":
                        "mm",

                    "cad_mapping": {
                        "source_frame":
                            "component_local",

                        "position":
                            component_reference_point[
                                "cad_local_position"
                            ]
                    }
                },

                "bounding_box":
                    bounding_box,

                "assembly_height":
                    None,

                # Explicit coordinate-frame diagnostics.
                "coordinate_frames": {
                    "component_local":
                        "component_local",

                    "occurrence_transform":
                        "component_local_to_fusion_root_assembly",

                    "assembly":
                        "fusion_root_assembly"
                }
            }

            assembly["parts"].append(
                part
            )

            # ----------------------------------------------
            # Save occurrence for relationship analysis.
            # ----------------------------------------------

            occurrences.append({
                "id":
                    occurrence.name,

                "bounding_box":
                    bounding_box
            })

        # ==================================================
        # Relationship detection
        # ==================================================

        geometric_relationships = (
            find_geometric_relationships(
                occurrences,
                tolerance=
                    GEOMETRIC_CONTACT_TOLERANCE_MM
            )
        )

        joint_relationships = (
            get_joint_relationships(
                root
            )
        )

        assembly["relationships"] = (
            merge_relationships(
                geometric_relationships,
                joint_relationships
            )
        )

        # ==================================================
        # Assembly axis
        # ==================================================

        axis_analysis = (
            analyze_global_assembly_axis(
                assembly["parts"],
                assembly["relationships"]
            )
        )

        assembly_axis = (
            axis_analysis[
                "selected_axis_vector"
            ]
        )

        assembly["assembly"][
            "assembly_axis"
        ] = assembly_axis

        assembly["assembly"][
            "axis_analysis"
        ] = axis_analysis

        # ==================================================
        # Add assembly heights now that the axis is known.
        # ==================================================

        height_map = get_part_height_map(
            assembly["parts"],
            assembly_axis
        )

        for part in assembly["parts"]:

            part["assembly_height"] = (
                height_map[
                    part["id"]
                ]
            )

        # ==================================================
        # Coordinate-frame / up-axis diagnostics
        # ==================================================

        component_frame_diagnostics = []

        for part in assembly["parts"]:

            rotation = part["rotation"]

            # Which component-local direction corresponds
            # to the global assembly axis?
            local_assembly_up = (
                get_assembly_axis_in_component(
                    rotation,
                    assembly_axis
                )
            )

            local_x_assembly = (
                get_component_axis_in_assembly(
                    rotation,
                    [1.0, 0.0, 0.0]
                )
            )

            local_y_assembly = (
                get_component_axis_in_assembly(
                    rotation,
                    [0.0, 1.0, 0.0]
                )
            )

            local_z_assembly = (
                get_component_axis_in_assembly(
                    rotation,
                    [0.0, 0.0, 1.0]
                )
            )

            yaw_analysis = (
                calculate_in_plane_yaw(
                    rotation,
                    assembly_axis,
                    local_assembly_up
                )
            )

            part[
                "coordinate_frames"
            ][
                "assembly_axis_in_component_local"
            ] = {
                "vector":
                    clean_vector(
                        local_assembly_up
                    ),

                "cardinal_axis":
                    classify_cardinal_axis(
                        local_assembly_up
                    )
            }

            part[
                "coordinate_frames"
            ][
                "component_axes_in_assembly"
            ] = {
                "local_x":
                    clean_vector(
                        local_x_assembly
                    ),

                "local_y":
                    clean_vector(
                        local_y_assembly
                    ),

                "local_z":
                    clean_vector(
                        local_z_assembly
                    )
            }

            part[
                "orientation"
            ] = {
                "full_rotation_matrix":
                    rotation,

                "in_plane_yaw":
                    yaw_analysis,

                "rotation_convention":
                    "component_local_to_fusion_root_assembly"
            }

            component_frame_diagnostics.append({
                "part":
                    part["id"],

                "component":
                    part["component"],

                "assembly_axis_in_component_local":
                    clean_vector(
                        local_assembly_up
                    ),

                "assembly_axis_cardinal_in_component":
                    classify_cardinal_axis(
                        local_assembly_up
                    ),

                "in_plane_yaw_deg":
                    yaw_analysis.get(
                        "degrees"
                    ),

                "rotation_matrix":
                    rotation
            })

        assembly[
            "coordinate_frame_analysis"
        ] = {
            "method":
                "explicit_occurrence_rotation_analysis",

            "assembly_axis":
                clean_vector(
                    assembly_axis
                ),

            "parts":
                component_frame_diagnostics,

            "note":
                "Do not assume component-local +Z is assembly up. The assembly axis is explicitly transformed into each component-local frame."
        }

        # ==================================================
        # Candidate directions
        # ==================================================

        assembly[
            "direction_analysis"
        ] = analyze_candidate_directions(
            assembly["parts"]
        )

        # ==================================================
        # Precedence
        # ==================================================

        assembly[
            "precedence_analysis"
        ] = analyze_joint_precedence(
            assembly["parts"],
            assembly["relationships"],
            assembly_axis
        )

        # ==================================================
        # Assembly plan
        # ==================================================

        assembly[
            "assembly_plan"
        ] = generate_assembly_plan(
            assembly["parts"],
            assembly["precedence_analysis"],
            ASSEMBLY_ANCHOR_OVERRIDE
        )

        # ==================================================
        # Validation diagnostics
        # ==================================================

        rotated_parts = []

        for part in assembly["parts"]:

            yaw = (
                part.get(
                    "orientation",
                    {}
                )
                .get(
                    "in_plane_yaw",
                    {}
                )
                .get(
                    "degrees"
                )
            )

            if (
                yaw is not None
                and abs(yaw) > 0.001
            ):

                rotated_parts.append(
                    part["id"]
                )

        assembly[
            "validation"
        ] = {
            "rotation_path": {
                "status":
                    (
                        "exercised"
                        if rotated_parts
                        else "not_exercised"
                    ),

                "rotated_parts":
                    rotated_parts,

                "note":
                    (
                        "At least one non-zero in-plane yaw was exported."
                        if rotated_parts
                        else
                        "No non-zero in-plane yaw was found. Test with a component rotated 45 or 90 degrees about the assembly axis."
                    )
            },

            "reference_point_policy": {
                "method":
                    "component_local_bounding_box_min_corner_default",

                "preserved":
                    True,

                "note":
                    "The bbox-min-corner remains the canonical component reference point unless explicitly overridden."
            },

            "mesh_policy": {
                "required":
                    True,

                "representation":
                    "3d_component_local_mesh",

                "note":
                    "Mesh generation is handled by toCV.py. AssemblyGuide stores the pose needed to place that local mesh."
            }
        }

        # ==================================================
        # JSON
        # ==================================================

        json_text = json.dumps(
            assembly,
            indent=4
        )

        # ==================================================
        # Save
        # ==================================================

        output_directory = os.path.expanduser(
            "~/PycharmProjects/Visionary/Fusion_output"
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        file_path = os.path.join(
            output_directory,
            "assembly.json"
        )

        with open(
            file_path,
            "w"
        ) as f:

            f.write(
                json_text
            )

        # ==================================================
        # Summary
        # ==================================================

        plan = assembly[
            "assembly_plan"
        ]

        if plan.get(
            "status"
        ) == "valid":

            plan_summary = (
                "\n"
                + "Plan status: valid"
                + "\n"
                + "Assembly steps: "
                + str(
                    plan.get(
                        "step_count",
                        0
                    )
                )
                + "\n"
                + "Operations: "
                + str(
                    plan.get(
                        "operation_count",
                        0
                    )
                )
            )

        else:

            plan_summary = (
                "\n"
                + "Plan status: ERROR"
                + "\n"
                + "Error: "
                + str(
                    plan.get(
                        "error",
                        "unknown"
                    )
                )
            )

        rotation_status = (
            assembly[
                "validation"
            ][
                "rotation_path"
            ][
                "status"
            ]
        )

        anchor = (
            plan.get(
                "assembly_anchor",
                {}
            )
            .get(
                "part",
                "none"
            )
        )

        # ==================================================
        # Confirmation
        # ==================================================

        ui.messageBox(
            "Export complete!\n\n"
            + "Parts: "
            + str(
                len(
                    assembly["parts"]
                )
            )
            + "\n"
            + "Components: "
            + str(
                len(
                    assembly["components"]
                )
            )
            + "\n"
            + "Relationships: "
            + str(
                len(
                    assembly["relationships"]
                )
            )
            + "\n"
            + "Selected axis: "
            + str(
                assembly[
                    "assembly"
                ][
                    "axis_analysis"
                ][
                    "selected_axis"
                ]
            )
            + "\n"
            + "Axis confidence: "
            + str(
                assembly[
                    "assembly"
                ][
                    "axis_analysis"
                ][
                    "confidence"
                ]
            )
            + "\n"
            + "Anchor: "
            + str(anchor)
            + "\n"
            + "Direction analyses: "
            + str(
                len(
                    assembly[
                        "direction_analysis"
                    ]
                )
            )
            + "\n"
            + "Precedence edges: "
            + str(
                len(
                    assembly[
                        "precedence_analysis"
                    ][
                        "precedence_edges"
                    ]
                )
            )
            + "\n"
            + "Rotation path: "
            + rotation_status
            + plan_summary
            + "\n\n"
            + "Saved to:\n"
            + file_path
        )

    except Exception as e:

        ui.messageBox(
            "ERROR:\n\n"
            + str(e)
        )