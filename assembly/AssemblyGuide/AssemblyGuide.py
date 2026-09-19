
import adsk.core
import adsk.fusion
import json
import os
import math


# Fusion's API stores design lengths in centimetres even when the UI
# displays millimetres. AssemblyGuide exports millimetres.
FUSION_CM_TO_MM = 10.0

# Preserve the original 0.1 cm broad-phase contact distance, but name it
# in the exported/planning coordinate unit used by this script.
GEOMETRIC_CONTACT_TOLERANCE_MM = 1.0

# Set this to a root occurrence name to override automatic anchor selection.
# Leave it as None to select the lowest legal root along the assembly axis.
ASSEMBLY_ANCHOR_OVERRIDE = None

# Optional component-specific reference points in millimetres. Each key is a
# Fusion component name and each value must contain a stable point in that
# component's local coordinate system. Use these for a deliberate CAD feature
# (for example, a mounting corner or insertion interface) instead of the
# default bounding-box corner.
#
# Example:
# COMPONENT_REFERENCE_POINT_OVERRIDES = {
#     "Base Plate": {
#         "name": "bottom_front_left_mounting_corner",
#         "local_position": [0.0, 0.0, 0.0]
#     }
# }
COMPONENT_REFERENCE_POINT_OVERRIDES = {}


# ==========================================================
# Utility functions
# ==========================================================

def clean_number(value):
    """Remove tiny floating-point errors and round values."""

    if abs(value) < 0.000001:
        return 0.0

    return round(value, 6)


def matrix_to_rotation(transform):
    """Extract the 3x3 rotation matrix from a Fusion transform."""

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
    """Extract XYZ position from a Fusion transform in millimetres."""

    return [
        clean_number(transform.getCell(0, 3) * FUSION_CM_TO_MM),
        clean_number(transform.getCell(1, 3) * FUSION_CM_TO_MM),
        clean_number(transform.getCell(2, 3) * FUSION_CM_TO_MM)
    ]


def point_to_list(point):
    """Convert a Fusion Point3D from centimetres to [x, y, z] mm."""

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
    Transform a component-local point into Fusion root assembly coordinates.

    All arguments and the returned point use millimetres. The occurrence
    position remains Fusion's arbitrary occurrence origin; this helper lets
    the exported plan instead use a meaningful component reference point.
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


def get_component_reference_point(
    component_id,
    component_bounds
):
    """
    Return the named local reference point for a component.

    The default is the minimum X/Y/Z corner of the component-local bounding
    box. It gives rectangular parts, such as plates, an intuitive and stable
    corner reference. An explicit override is available for irregular parts
    whose useful assembly reference is a CAD feature rather than a box corner.
    """

    override = COMPONENT_REFERENCE_POINT_OVERRIDES.get(
        component_id
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
                "method": "configured_component_reference",
                "cad_local_position": [
                    clean_number(value)
                    for value in local_position
                ],
                "local_coordinate_frame": "component_local",
                "units": "mm"
            }

        raise ValueError(
            "Reference-point override for '"
            + component_id
            + "' must contain a three-value local_position."
        )

    if component_bounds is None:

        return {
            "name": "component_local_origin",
            "method": "component_bounds_unavailable",
            "cad_local_position": [0.0, 0.0, 0.0],
            "local_coordinate_frame": "component_local",
            "units": "mm"
        }

    return {
        "name": "component_local_bounding_box_min_corner",
        "method": "component_local_bounding_box_minimum",
        "cad_local_position": component_bounds["min"],
        "local_coordinate_frame": "component_local",
        "units": "mm"
    }


# ==========================================================
# Bounding boxes
# ==========================================================

def get_bounding_box(occurrence):
    """Get the occurrence bounding box in assembly coordinates."""

    box = occurrence.boundingBox

    return {
        "min": point_to_list(box.minPoint),
        "max": point_to_list(box.maxPoint)
    }


def get_component_bounds(component):
    """
    Get component geometry bounds.

    These coordinates are in the component's own coordinate
    system.
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

        min_x = min(min_x, box.minPoint.x)
        min_y = min(min_y, box.minPoint.y)
        min_z = min(min_z, box.minPoint.z)

        max_x = max(max_x, box.maxPoint.x)
        max_y = max(max_y, box.maxPoint.y)
        max_z = max(max_z, box.maxPoint.z)

    return {
        "min": [
            clean_number(min_x * FUSION_CM_TO_MM),
            clean_number(min_y * FUSION_CM_TO_MM),
            clean_number(min_z * FUSION_CM_TO_MM)
        ],
        "max": [
            clean_number(max_x * FUSION_CM_TO_MM),
            clean_number(max_y * FUSION_CM_TO_MM),
            clean_number(max_z * FUSION_CM_TO_MM)
        ]
    }


# ==========================================================
# Geometry relationship detection
# ==========================================================

def bounding_boxes_close(box_a, box_b, tolerance=0.1):
    """
    Determine whether two bounding boxes overlap or are
    within the specified tolerance.

    This is NOT exact contact detection. It is a broad-phase
    geometric relationship.
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


def find_geometric_relationships(occurrences, tolerance=0.1):
    """
    Find candidate geometric relationships between every
    pair of occurrences.
    """

    relationships = []

    for i in range(len(occurrences)):

        for j in range(i + 1, len(occurrences)):

            a = occurrences[i]
            b = occurrences[j]

            if bounding_boxes_close(
                a["bounding_box"],
                b["bounding_box"],
                tolerance
            ):

                relationships.append({
                    "part_a": a["id"],
                    "part_b": b["id"],
                    "type": "candidate_contact",
                    "confidence": "low"
                })

    return relationships


# ==========================================================
# Fusion joint extraction
# ==========================================================

def get_joint_relationships(root):
    """
    Extract Fusion joints associated with occurrences.

    Uses occurrence.joints so the exporter remains independent
    of newer AssemblyConstraints APIs.
    """

    relationships = []
    seen = set()

    for occurrence in root.occurrences:

        try:
            joints = occurrence.joints

        except Exception:
            continue

        for i in range(joints.count):

            try:
                joint = joints.item(i)

            except Exception:
                continue

            if not joint:
                continue

            occurrence_one = None
            occurrence_two = None

            try:
                occurrence_one = joint.occurrenceOne

            except Exception:
                pass

            try:
                occurrence_two = joint.occurrenceTwo

            except Exception:
                pass

            if not occurrence_one or not occurrence_two:
                continue

            id_one = occurrence_one.name
            id_two = occurrence_two.name

            pair_key = tuple(sorted([
                id_one,
                id_two
            ]))

            if pair_key in seen:
                continue

            seen.add(pair_key)

            joint_type = "unknown"

            try:
                joint_type = str(joint.objectType)

            except Exception:
                pass

            relationships.append({
                "part_a": id_one,
                "part_b": id_two,
                "type": "joint",
                "joint_type": joint_type,
                "confidence": "high"
            })

    return relationships


# ==========================================================
# Merge relationship evidence
# ==========================================================

def merge_relationships(
    geometric_relationships,
    joint_relationships
):
    """
    Combine geometric and CAD relationship evidence.

    A pair can have both candidate_contact and joint evidence.
    """

    relationship_map = {}

    # ------------------------------------------------------
    # Geometric relationships
    # ------------------------------------------------------

    for relationship in geometric_relationships:

        key = tuple(sorted([
            relationship["part_a"],
            relationship["part_b"]
        ]))

        relationship_map[key] = {
            "part_a": relationship["part_a"],
            "part_b": relationship["part_b"],
            "evidence": [
                {
                    "type": "candidate_contact",
                    "confidence": "low"
                }
            ]
        }

    # ------------------------------------------------------
    # Fusion joint relationships
    # ------------------------------------------------------

    for relationship in joint_relationships:

        key = tuple(sorted([
            relationship["part_a"],
            relationship["part_b"]
        ]))

        if key not in relationship_map:

            relationship_map[key] = {
                "part_a": relationship["part_a"],
                "part_b": relationship["part_b"],
                "evidence": []
            }

        relationship_map[key]["evidence"].append({
            "type": "joint",
            "joint_type": relationship["joint_type"],
            "confidence": "high"
        })

    return list(relationship_map.values())


# ==========================================================
# Assembly axis identification
# ==========================================================

def get_bbox_dimensions(box):
    """Return [x, y, z] dimensions for an AABB."""

    return [
        max(0.0, box["max"][0] - box["min"][0]),
        max(0.0, box["max"][1] - box["min"][1]),
        max(0.0, box["max"][2] - box["min"][2])
    ]


def get_axis_interval_metrics(box_a, box_b, axis_index):
    """
    Compare two bounding boxes along one candidate assembly axis.

    The important distinction here is that we do NOT use the
    center-to-center vector as the axis signal.

    Instead, this measures:

        1. overlap along the candidate axis
        2. separation along the candidate axis
        3. overlap on the two perpendicular axes

    A stacking/interface relationship therefore tends to have:

        low overlap along the assembly axis
        high overlap in the perpendicular directions
    """

    dimensions_a = get_bbox_dimensions(box_a)
    dimensions_b = get_bbox_dimensions(box_b)

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
            min(a_max, b_max) - max(a_min, b_min)
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
                    overlap / reference_dimension
                )
            )

        else:

            overlap_ratio = 0.0

        overlaps.append(overlap)
        gaps.append(gap)
        overlap_ratios.append(overlap_ratio)

    axis_overlap_ratio = overlap_ratios[axis_index]

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
        1.0 - axis_overlap_ratio
    )

    interface_score = (
        separation_score
        * perpendicular_overlap
    )

    return {
        "overlap": clean_number(
            overlaps[axis_index]
        ),

        "gap": clean_number(
            gaps[axis_index]
        ),

        "overlap_ratio": clean_number(
            axis_overlap_ratio
        ),

        "separation_score": clean_number(
            separation_score
        ),

        "perpendicular_overlap": clean_number(
            perpendicular_overlap
        ),

        "interface_score": clean_number(
            interface_score
        ),

        "all_axis_overlap_ratios": [
            clean_number(value)
            for value in overlap_ratios
        ]
    }


def get_relationship_weight(relationship):
    """
    Return the amount of trust given to a relationship.

    Fusion joints are strong CAD evidence.

    Candidate geometric contacts are intentionally weaker so
    that a coincidental contact between long parts cannot
    overpower several explicit joints.
    """

    has_joint = False
    has_candidate_contact = False

    for evidence in relationship.get(
        "evidence",
        []
    ):

        if evidence.get("type") == "joint":

            has_joint = True

        elif evidence.get("type") == "candidate_contact":

            has_candidate_contact = True

    if has_joint:
        return 2.0

    if has_candidate_contact:
        return 0.5

    return 0.0


def get_relationship_evidence_types(relationship):
    """Return the evidence types present in one relationship."""

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
    Identify the primary assembly axis from CAD relationships.

    The axis is selected as an undirected X/Y/Z axis first.
    The sign (+/-) is determined afterward from the
    lower-to-higher direction of the related parts.

    Primary evidence:

        - AABB interface/stacking geometry
        - high-confidence Fusion joints

    Deliberately NOT used as the primary axis signal:

        - center-to-center displacement
        - motion/removability simulation
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

            metrics = get_axis_interval_metrics(
                part_a["bounding_box"],
                part_b["bounding_box"],
                axis_index
            )

            axis_metrics.append(
                metrics
            )

            axis_totals[axis_index] += (
                weight
                * metrics["interface_score"]
            )

            axis_weights[axis_index] += weight

        center_a = part_a["position"]
        center_b = part_b["position"]

        center_difference = [
            center_b[i] - center_a[i]
            for i in range(3)
        ]

        relationship_diagnostics.append({
            "part_a": part_a["id"],
            "part_b": part_b["id"],

            "evidence":
                get_relationship_evidence_types(
                    relationship
                ),

            "weight":
                clean_number(weight),

            "interface_geometry": {
                "X": axis_metrics[0],
                "Y": axis_metrics[1],
                "Z": axis_metrics[2]
            },

            "center_difference_diagnostic": [
                clean_number(value)
                for value in center_difference
            ]
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
        for relationship in relationship_diagnostics
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
            key=lambda index: axis_scores[index]
        )

        sorted_scores = sorted(
            axis_scores,
            reverse=True
        )

        if len(sorted_scores) >= 2:

            margin = (
                sorted_scores[0]
                - sorted_scores[1]
            )

        else:

            margin = sorted_scores[0]

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

    # ------------------------------------------------------
    # Determine sign separately.
    # ------------------------------------------------------

    signed_direction = 0.0

    selected_axis_index = selected_index

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
            part_b["position"][selected_axis_index]
            - part_a["position"][selected_axis_index]
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
            "X": axis_scores[0],
            "Y": axis_scores[1],
            "Z": axis_scores[2]
        },

        "axis_weights": {
            "X": clean_number(
                axis_weights[0]
            ),
            "Y": clean_number(
                axis_weights[1]
            ),
            "Z": clean_number(
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
            len(relationship_diagnostics),

        "high_confidence_relationship_count":
            high_confidence_relationship_count,

        "relationship_analysis":
            relationship_diagnostics
    }


# ==========================================================
# Candidate direction generation
# ==========================================================

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

    if length == 0:

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


def add_direction(
    directions,
    name,
    vector,
    frame,
    source
):
    """
    Add a direction unless an equivalent direction
    already exists.
    """

    vector = normalize_vector(
        vector
    )

    for existing in directions:

        if vectors_are_close(
            existing["vector"],
            vector
        ):

            return

    directions.append({
        "name": name,

        "vector": [
            clean_number(vector[0]),
            clean_number(vector[1]),
            clean_number(vector[2])
        ],

        "frame": frame,

        "source": source
    })


def generate_candidate_directions(
    rotation
):
    """
    Generate candidate assembly/disassembly directions
    using global and local coordinate axes.

    This stage generates candidate directions only.
    It does not determine whether a direction is
    geometrically feasible.
    """

    directions = []

    # ------------------------------------------------------
    # GLOBAL AXES
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # LOCAL AXES
    # ------------------------------------------------------

    local_x = [
        rotation[0][0],
        rotation[1][0],
        rotation[2][0]
    ]

    local_y = [
        rotation[0][1],
        rotation[1][1],
        rotation[2][1]
    ]

    local_z = [
        rotation[0][2],
        rotation[1][2],
        rotation[2][2]
    ]

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
    """
    Generate candidate directions for every part.

    This stage does NOT determine geometric feasibility yet.
    It only determines which directions should be tested.
    """

    results = []

    for part in parts:

        directions = generate_candidate_directions(
            part["rotation"]
        )

        results.append({
            "part": part["id"],
            "candidate_directions": directions
        })

    return results


# ==========================================================
# Joint graph and precedence analysis
# ==========================================================

def get_joint_graph(relationships):
    """
    Build an undirected graph from high-confidence Fusion joint
    relationships.

    The graph represents physical/CAD connectivity, not assembly
    order.
    """

    graph = {}

    for relationship in relationships:

        has_joint_evidence = False

        for evidence in relationship.get(
            "evidence",
            []
        ):

            if evidence.get("type") == "joint":

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
            graph[part_a].append(part_b)

        if part_a not in graph[part_b]:
            graph[part_b].append(part_a)

    return graph


def get_part_height_map(
    parts,
    assembly_axis
):
    """
    Calculate each part's scalar position along the assembly axis.

    For +Z, use the bottom of the full occurrence bounding box.

    For other axes, use the occurrence position projected onto
    the selected axis.
    """

    axis = normalize_vector(
        assembly_axis
    )

    heights = {}

    for part in parts:

        if vectors_are_close(
            axis,
            [0, 0, 1]
        ):

            height = part.get(
                "assembly_height",
                None
            )

            if height is None:

                position = part["position"]

                height = (
                    position[0] * axis[0]
                    + position[1] * axis[1]
                    + position[2] * axis[2]
                )

        else:

            position = part["position"]

            height = (
                position[0] * axis[0]
                + position[1] * axis[1]
                + position[2] * axis[2]
            )

        heights[part["id"]] = clean_number(
            height
        )

    return heights


def make_precedence_edge(
    before,
    after,
    height_before,
    height_after,
    reason
):
    """Create one directed precedence edge."""

    return {
        "before": before,
        "after": after,
        "reason": reason,
        "evidence": "fusion_joint",
        "confidence": "high",
        "height_before": clean_number(
            height_before
        ),
        "height_after": clean_number(
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
    Prepare precedence information from the Fusion-joint graph.

    A joint connecting a lower part to a higher part gives:

        lower part -> higher part

    Equal-height pairs remain ambiguous.

    This produces a partial order rather than forcing a
    single sequence.
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
                height_b - height_a
            )

            if abs(
                height_difference
            ) <= height_tolerance:

                ambiguous_pairs.append({
                    "part_a": part_a,
                    "part_b": part_b,
                    "reason": "same_assembly_height",
                    "evidence": "fusion_joint",
                    "confidence": "high"
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

    # ------------------------------------------------------
    # Build directed adjacency.
    # ------------------------------------------------------

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

        if after not in precedence_graph[before]:

            precedence_graph[before].append(
                after
            )

    # ------------------------------------------------------
    # Identify roots and leaves.
    # ------------------------------------------------------

    incoming_count = {}

    for part in precedence_graph:
        incoming_count[part] = 0

    for before in precedence_graph:

        for after in precedence_graph[before]:

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

    # ------------------------------------------------------
    # Find connected components.
    # ------------------------------------------------------

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

        "joint_graph":
            joint_graph,

        "part_heights":
            heights,

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
# Assembly plan generation
# ==========================================================

def transpose_rotation(rotation):
    """Return the transpose/inverse of an orthonormal 3x3 rotation."""

    return [
        [rotation[column][row] for column in range(3)]
        for row in range(3)
    ]


def multiply_rotation_matrices(left, right):
    """Return the product of two 3x3 rotation matrices."""

    return [
        [
            clean_number(
                sum(
                    left[row][index] * right[index][column]
                    for index in range(3)
                )
            )
            for column in range(3)
        ]
        for row in range(3)
    ]


def multiply_rotation_vector(rotation, vector):
    """Return a 3x3 rotation matrix multiplied by a 3D vector."""

    return [
        clean_number(
            sum(
                rotation[row][column] * vector[column]
                for column in range(3)
            )
        )
        for row in range(3)
    ]


def get_part_reference_position(part):
    """Return the assembly-space point used to position this part in a plan."""

    reference_point = part.get(
        "reference_point",
        {}
    )

    return reference_point.get(
        "assembly_position",
        part["position"]
    )


def pose_relative_to_anchor(anchor_part, target_part):
    """
    Express a target part's final CAD pose in the anchor part's local frame.

    At runtime, Visionary can combine this fixed relative pose with the
    observed anchor pose. The whole assembly can then shift or rotate on the
    workbench while subsequent targets stay correct relative to the anchor.
    """

    anchor_rotation_inverse = transpose_rotation(
        anchor_part["rotation"]
    )

    anchor_position = get_part_reference_position(
        anchor_part
    )

    target_position = get_part_reference_position(
        target_part
    )

    position_delta = [
        target_position[axis]
        - anchor_position[axis]
        for axis in range(3)
    ]

    return {
        "position": multiply_rotation_vector(
            anchor_rotation_inverse,
            position_delta
        ),

        "rotation": multiply_rotation_matrices(
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
    Select the component that establishes the live assembly reference frame.

    Only roots of the precedence graph are eligible because the anchor must be
    placeable before every operation that is expressed relative to it. When
    multiple roots are legal, select the lowest along the assembly axis and
    use the occurrence name as a deterministic tie-breaker.
    """

    candidate_ids = sorted([
        part_id
        for part_id, count in incoming_count.items()
        if count == 0 and part_id in part_by_id
    ])

    if not candidate_ids:

        return {
            "status": "no_anchor_candidate",
            "part": None,
            "candidates": []
        }

    part_heights = precedence_analysis.get(
        "part_heights",
        {}
    )

    candidates = [
        {
            "part": part_id,
            "assembly_axis_projection_mm": clean_number(
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
                "status": "invalid_anchor_override",
                "part": None,
                "requested_part": anchor_override,
                "candidates": candidates
            }

        return {
            "status": "configured",
            "method": "configured_root_override",
            "part": anchor_override,
            "candidates": candidates
        }

    selected_part = min(
        candidate_ids,
        key=lambda part_id: (
            part_heights.get(part_id, 0.0),
            part_id
        )
    )

    return {
        "status": "automatic",
        "method": "root_lowest_along_assembly_axis",
        "part": selected_part,
        "candidates": candidates,
        "tie_breaker": "occurrence_name_ascending"
    }


def generate_assembly_plan(
    parts,
    precedence_analysis,
    anchor_override=None
):
    """
    Generate a staged assembly plan from the precedence graph.

    The planner performs a topological-style staged traversal.

    At each stage:

        1. Find all parts whose dependencies are complete.
        2. Put all such independent parts into the same step.
        3. Mark those parts as completed.
        4. Repeat until every part is planned.

    This means the planner does NOT invent an ordering between
    independent parts.

    Example:

        Plate1 ──┐
                 ├──> Engine ──> Beam
        Plate2 ──┘

    becomes:

        Step 1: Plate1, Plate2
        Step 2: Engine
        Step 3: Beam

    Every part is currently represented as a PLACE operation with
    its final CAD target pose. This lets the projector and future
    state machine consume the same expected position and rotation
    without inventing target locations at runtime.

    Special operations such as FASTEN, INSERT, SNAP, etc.
    can be added later without changing the core planning
    algorithm.
    """

    precedence_graph = precedence_analysis.get(
        "precedence_graph",
        {}
    )

    # ------------------------------------------------------
    # Make sure every exported part exists in the planning
    # graph.
    #
    # Parts with no Fusion-joint relationships would otherwise
    # be invisible to the precedence graph.
    # ------------------------------------------------------

    all_part_ids = [
        part["id"]
        for part in parts
    ]

    graph = {}

    part_by_id = {
        part["id"]: part
        for part in parts
    }

    for part_id in all_part_ids:

        graph[part_id] = []

    for before in precedence_graph:

        if before not in graph:
            graph[before] = []

        for after in precedence_graph[before]:

            if after not in graph:
                graph[after] = []

            if after not in graph[before]:

                graph[before].append(
                    after
                )

    # ------------------------------------------------------
    # Build incoming dependency counts.
    # ------------------------------------------------------

    incoming_count = {
        part_id: 0
        for part_id in graph
    }

    for before in graph:

        for after in graph[before]:

            incoming_count[after] += 1

    # ------------------------------------------------------
    # Select an anchor that can be placed in the first stage.
    # ------------------------------------------------------

    anchor_selection = select_assembly_anchor(
        part_by_id,
        incoming_count,
        precedence_analysis,
        anchor_override
    )

    if anchor_selection["part"] is None:

        if not anchor_selection["candidates"] and graph:

            return {
                "method": "precedence_graph_topological_planning",
                "status": "error",
                "error": "precedence_graph_cycle",
                "unresolved_parts": sorted(graph.keys()),
                "steps": []
            }

        return {
            "method": "precedence_graph_topological_planning",
            "status": "error",
            "error": anchor_selection["status"],
            "anchor_selection": anchor_selection,
            "steps": []
        }

    anchor_part = part_by_id[
        anchor_selection["part"]
    ]

    # ------------------------------------------------------
    # Planning state.
    # ------------------------------------------------------

    remaining = set(
        graph.keys()
    )

    completed = set()

    steps = []

    step_number = 1

    # ------------------------------------------------------
    # Repeatedly find currently available parts.
    # ------------------------------------------------------

    while remaining:

        available = []

        for part_id in sorted(
            remaining
        ):

            dependencies_satisfied = True

            for dependency in graph:

                if part_id in graph[dependency]:

                    if dependency not in completed:

                        dependencies_satisfied = False
                        break

            if dependencies_satisfied:

                available.append(
                    part_id
                )

        # --------------------------------------------------
        # No available part means the precedence graph
        # contains a cycle.
        # --------------------------------------------------

        if not available:

            cycle_parts = sorted(
                remaining
            )

            return {
                "method":
                    "precedence_graph_topological_planning",

                "status":
                    "error",

                "error":
                    "precedence_graph_cycle",

                "unresolved_parts":
                    cycle_parts,

                "steps":
                    []
            }

        # --------------------------------------------------
        # Create operations for this step.
        # --------------------------------------------------

        operations = []

        for part_id in available:

            part = part_by_id[part_id]

            dependencies = sorted([
                dependency
                for dependency in graph
                if part_id in graph[dependency]
            ])

            relative_pose = pose_relative_to_anchor(
                anchor_part,
                part
            )

            reference_point = part.get(
                "reference_point",
                {}
            )

            operations.append({
                "id": "place_" + part_id,

                "type": "PLACE",

                "part": part_id,

                "component": part["component"],

                "target_position": get_part_reference_position(
                    part
                ),

                "target_position_reference": {
                    "name": reference_point.get(
                        "name",
                        "occurrence_origin"
                    ),
                    "method": reference_point.get(
                        "method",
                        "legacy_occurrence_origin"
                    ),
                    "reference_frame": reference_point.get(
                        "reference_frame",
                        "occurrence_origin_frame"
                    ),
                    "reference_frame_position": [0.0, 0.0, 0.0]
                },

                "target_rotation": part["rotation"],

                "target_coordinate_frame":
                    "fusion_root_assembly",

                "target_units": "mm",

                "target_relative_to_anchor": {
                    "anchor_part": anchor_part["id"],
                    "position": relative_pose["position"],
                    "rotation": relative_pose["rotation"],
                    "coordinate_frame":
                        "anchor_reference_point_local",
                    "position_units": "mm"
                },

                "dependencies": [
                    "place_" + dependency
                    for dependency in dependencies
                ],

                "depends_on_parts": dependencies
            })

        steps.append({
            "step": step_number,
            "operations": operations
        })

        # --------------------------------------------------
        # Mark all operations in this stage complete from
        # the planner's perspective.
        #
        # This does NOT mean the human has completed them yet.
        # The runtime state machine will track actual completion.
        # --------------------------------------------------

        for part_id in available:

            remaining.remove(
                part_id
            )

            completed.add(
                part_id
            )

        step_number += 1

    # ------------------------------------------------------
    # Return the generated plan.
    # ------------------------------------------------------

    return {
        "method":
            "precedence_graph_topological_planning",

        "status":
            "valid",

        "coordinate_system": {
            "frame": "fusion_root_assembly",
            "units": "mm",
            "origin": "Fusion root component origin"
        },

        "assembly_anchor": {
            "part": anchor_part["id"],
            "operation": "place_" + anchor_part["id"],
            "reference_point": anchor_part.get(
                "reference_point",
                {
                    "name": "occurrence_origin",
                    "assembly_position": anchor_part["position"],
                    "units": "mm"
                }
            ),
            "selection": anchor_selection
        },

        "steps":
            steps,

        "operation_count":
            sum(
                len(step["operations"])
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
        # Get active design
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
        # Assembly configuration
        # ==================================================

        assembly = {
            "assembly": {
                "name": root.name,
                "units": "mm"
            },

            "components": [],

            "parts": [],

            "relationships": [],

            "direction_analysis": [],

            "precedence_analysis": {},

            "assembly_plan": {}
        }

        # ==================================================
        # Track unique component definitions
        # ==================================================

        components_seen = set()

        component_bounds_by_id = {}

        # ==================================================
        # Store occurrence information for analysis
        # ==================================================

        occurrences = []

        # ==================================================
        # Export occurrences
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

            component_id = occurrence.component.name

            # ----------------------------------------------
            # Component definition
            # ----------------------------------------------

            if component_id not in components_seen:

                component = occurrence.component

                component_bounds = get_component_bounds(
                    component
                )

                component_reference_point = (
                    get_component_reference_point(
                        component_id,
                        component_bounds
                    )
                )

                assembly["components"].append({
                    "id": component_id,

                    "bounding_box":
                        component_bounds,

                    "reference_point": {
                        "name": component_reference_point["name"],
                        "method": component_reference_point["method"],
                        "reference_frame":
                            "component_reference_point_local",
                        "reference_frame_position": [0.0, 0.0, 0.0],
                        "units": "mm",
                        "cad_mapping": {
                            "source_frame": "component_local",
                            "position": component_reference_point[
                                "cad_local_position"
                            ]
                        }
                    }
                })

                component_bounds_by_id[component_id] = (
                    component_bounds
                )

                components_seen.add(
                    component_id
                )

            component_reference_point = (
                get_component_reference_point(
                    component_id,
                    component_bounds_by_id[component_id]
                )
            )

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
            # Assembly height
            # ----------------------------------------------

            assembly_height = (
                bounding_box["min"][2]
            )

            # ----------------------------------------------
            # Part
            # ----------------------------------------------

            part = {
                "id":
                    occurrence.name,

                "component":
                    component_id,

                "position":
                    position,

                "occurrence_origin": position,

                "rotation":
                    rotation,

                "reference_point": {
                    "name": component_reference_point["name"],
                    "method": component_reference_point["method"],
                    "reference_frame": "part_reference_point_local",
                    "reference_frame_position": [0.0, 0.0, 0.0],
                    "assembly_position": reference_position,
                    "coordinate_frame": "fusion_root_assembly",
                    "units": "mm",
                    "cad_mapping": {
                        "source_frame": "component_local",
                        "position": component_reference_point[
                            "cad_local_position"
                        ]
                    }
                },

                "bounding_box":
                    bounding_box,

                "assembly_height":
                    clean_number(
                        assembly_height
                    )
            }

            assembly["parts"].append(
                part
            )

            # ----------------------------------------------
            # Save occurrence for relationship analysis
            # ----------------------------------------------

            occurrences.append({
                "id":
                    occurrence.name,

                "bounding_box":
                    bounding_box
            })

        # ==================================================
        # Geometric relationships
        # ==================================================

        geometric_relationships = (
            find_geometric_relationships(
                occurrences,
                tolerance=GEOMETRIC_CONTACT_TOLERANCE_MM
            )
        )

        # ==================================================
        # Fusion joint relationships
        # ==================================================

        joint_relationships = (
            get_joint_relationships(
                root
            )
        )

        # ==================================================
        # Merge relationship evidence
        # ==================================================

        assembly["relationships"] = (
            merge_relationships(
                geometric_relationships,
                joint_relationships
            )
        )

        # ==================================================
        # Assembly axis identification
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
        # Candidate direction generation
        # ==================================================

        assembly["direction_analysis"] = (
            analyze_candidate_directions(
                assembly["parts"]
            )
        )

        # ==================================================
        # Precedence analysis
        # ==================================================

        assembly["precedence_analysis"] = (
            analyze_joint_precedence(
                assembly["parts"],
                assembly["relationships"],
                assembly_axis
            )
        )

        # ==================================================
        # Automatic assembly plan generation
        # ==================================================

        assembly["assembly_plan"] = (
            generate_assembly_plan(
                assembly["parts"],
                assembly["precedence_analysis"],
                ASSEMBLY_ANCHOR_OVERRIDE
            )
        )

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
            "~/projects/Fusion API/AssemblyGuide"
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
        # Plan summary
        # ==================================================

        plan = assembly["assembly_plan"]

        if plan.get("status") == "valid":

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

        # ==================================================
        # Confirmation
        # ==================================================

        ui.messageBox(
            "Export complete!\n\n"
            + "Parts: "
            + str(
                len(assembly["parts"])
            )
            + "\n"
            + "Relationships: "
            + str(
                len(assembly["relationships"])
            )
            + "\n"
            + "Selected axis: "
            + str(
                assembly["assembly"][
                    "axis_analysis"
                ]["selected_axis"]
            )
            + "\n"
            + "Axis confidence: "
            + str(
                assembly["assembly"][
                    "axis_analysis"
                ]["confidence"]
            )
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
