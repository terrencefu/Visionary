import adsk.core
import adsk.fusion
import json
import os
import math


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
    """Extract XYZ position from a Fusion transform."""

    return [
        clean_number(transform.getCell(0, 3)),
        clean_number(transform.getCell(1, 3)),
        clean_number(transform.getCell(2, 3))
    ]


def point_to_list(point):
    """Convert a Fusion Point3D to [x, y, z]."""

    return [
        clean_number(point.x),
        clean_number(point.y),
        clean_number(point.z)
    ]


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
            clean_number(min_x),
            clean_number(min_y),
            clean_number(min_z)
        ],
        "max": [
            clean_number(max_x),
            clean_number(max_y),
            clean_number(max_z)
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
        return [0.0, 0.0, 0.0]

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
            vector_a[i] - vector_b[i]
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

    vector = normalize_vector(vector)

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
    Generate candidate directions for individual-part motion
    analysis.

    Global directions:
        +/- X
        +/- Y
        +/- Z

    Local directions:
        +/- x
        +/- y
        +/- z

    This function is separate from global assembly-axis
    identification.

    Global assembly-axis identification does NOT use these
    motion directions.
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
    # Rotation matrix columns represent the local X/Y/Z axes
    # expressed in global coordinates.

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
    It only determines which directions should be tested for
    individual-part motion.
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
# Global assembly-axis identification
# ==========================================================

def get_relationship_weight(relationship):
    """
    Assign a weight to a relationship based on the strength of
    its CAD evidence.

    A Fusion joint is stronger evidence than broad-phase
    candidate contact.

    If a pair has both types of evidence, the relationship gets
    a maximum weight of 3.0 rather than being allowed to dominate
    the complete assembly.
    """

    weight = 0.0

    for evidence in relationship.get("evidence", []):

        evidence_type = evidence.get("type")

        if evidence_type == "joint":
            weight += 2.0

        elif evidence_type == "candidate_contact":
            weight += 1.0

    return min(weight, 3.0)


def get_part_map(parts):
    """Create a lookup from part ID to part data."""

    return {
        part["id"]: part
        for part in parts
    }


def get_part_center(part):
    """
    Get the center of an occurrence's assembly-coordinate
    bounding box.

    The bounding-box center is used instead of the occurrence
    transform origin because the origin does not necessarily
    represent the physical center of the part.
    """

    box = part["bounding_box"]

    return [
        (
            box["min"][0]
            + box["max"][0]
        ) / 2.0,

        (
            box["min"][1]
            + box["max"][1]
        ) / 2.0,

        (
            box["min"][2]
            + box["max"][2]
        ) / 2.0
    ]


def get_position_difference(
    part_a,
    part_b
):
    """
    Return the vector from part A's bounding-box center to
    part B's bounding-box center.
    """

    center_a = get_part_center(part_a)
    center_b = get_part_center(part_b)

    return [
        center_b[0] - center_a[0],
        center_b[1] - center_a[1],
        center_b[2] - center_a[2]
    ]


def calculate_relationship_axis_alignment(
    part_a,
    part_b,
    candidate_axis
):
    """
    Measure how strongly the relationship between two parts
    aligns with an UN-DIRECTED candidate axis.

    Returns:

        1.0 = relationship lies completely along the axis
        0.0 = relationship is perpendicular to the axis

    Absolute value is intentional.

    Axis identification should determine:

        X vs Y vs Z

    rather than prematurely deciding:

        +X vs -X
        +Y vs -Y
        +Z vs -Z
    """

    difference = get_position_difference(
        part_a,
        part_b
    )

    distance = vector_length(
        difference
    )

    if distance < 0.000001:
        return 0.0

    difference_direction = normalize_vector(
        difference
    )

    projection = (
        difference_direction[0] * candidate_axis[0]
        + difference_direction[1] * candidate_axis[1]
        + difference_direction[2] * candidate_axis[2]
    )

    return abs(projection)


def calculate_relationship_separation(
    part_a,
    part_b,
    candidate_axis
):
    """
    Measure how much the centers of two related parts are
    separated along a candidate axis.

    This is normalized relative to the center-to-center
    distance, so the result lies approximately in [0, 1].

    This is related to axis alignment, but is retained as a
    separate interpretable signal for the holistic analysis.
    """

    difference = get_position_difference(
        part_a,
        part_b
    )

    distance = vector_length(
        difference
    )

    if distance < 0.000001:
        return 0.0

    projected_distance = abs(
        difference[0] * candidate_axis[0]
        + difference[1] * candidate_axis[1]
        + difference[2] * candidate_axis[2]
    )

    return min(
        1.0,
        projected_distance / distance
    )


def calculate_same_layer_fraction(
    parts,
    relationships,
    candidate_axis,
    layer_tolerance=0.01
):
    """
    Determine what fraction of related pairs occupy essentially
    the same layer along the candidate axis.

    A high same-layer fraction weakens the case for that axis
    because related parts are not being separated into a useful
    progression along it.
    """

    part_map = get_part_map(parts)

    weighted_pairs = 0.0
    weighted_same_layer = 0.0

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

        if weight <= 0:
            continue

        center_a = get_part_center(part_a)
        center_b = get_part_center(part_b)

        projected_a = (
            center_a[0] * candidate_axis[0]
            + center_a[1] * candidate_axis[1]
            + center_a[2] * candidate_axis[2]
        )

        projected_b = (
            center_b[0] * candidate_axis[0]
            + center_b[1] * candidate_axis[1]
            + center_b[2] * candidate_axis[2]
        )

        separation = abs(
            projected_b - projected_a
        )

        weighted_pairs += weight

        if separation <= layer_tolerance:
            weighted_same_layer += weight

    if weighted_pairs <= 0:
        return 0.0

    return (
        weighted_same_layer
        / weighted_pairs
    )


def calculate_spatial_distribution_score(
    parts,
    candidate_axis
):
    """
    Measure how much of the assembly's overall spatial extent lies
    along the candidate axis.

    This is a secondary holistic signal.

    It is deliberately weaker than relationship evidence because
    a product can be physically wide without being assembled along
    its widest dimension.
    """

    if len(parts) <= 1:
        return {
            "score": 0.0,
            "axis_extent": 0.0,
            "total_extent": 0.0
        }

    projected_positions = []

    for part in parts:

        center = get_part_center(part)

        projection = (
            center[0] * candidate_axis[0]
            + center[1] * candidate_axis[1]
            + center[2] * candidate_axis[2]
        )

        projected_positions.append(
            projection
        )

    axis_extent = (
        max(projected_positions)
        - min(projected_positions)
    )

    # Calculate the overall positional spread in XYZ.
    all_centers = [
        get_part_center(part)
        for part in parts
    ]

    extents = []

    for axis_index in range(3):

        values = [
            center[axis_index]
            for center in all_centers
        ]

        extents.append(
            max(values) - min(values)
        )

    total_extent = vector_length(
        extents
    )

    if total_extent < 0.000001:

        score = 0.0

    else:

        score = (
            axis_extent / total_extent
        )

    return {
        "score": clean_number(score),
        "axis_extent": clean_number(axis_extent),
        "total_extent": clean_number(total_extent)
    }


def calculate_assembly_axis_score(
    parts,
    relationships,
    candidate_name,
    candidate_axis
):
    """
    Calculate the holistic score for one UN-DIRECTED global
    assembly axis.

    Candidate axes are:

        X
        Y
        Z

    The sign is deliberately ignored during this stage.

    Signals:

        1. Relationship alignment
        2. Relationship separation
        3. Same-layer penalty
        4. Overall spatial distribution

    Relationship evidence is the primary signal.

    Spatial distribution is only supporting evidence.
    """

    part_map = get_part_map(parts)

    total_weight = 0.0

    weighted_alignment = 0.0
    weighted_separation = 0.0

    supporting_pairs = []
    perpendicular_pairs = []

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

        if weight <= 0:
            continue

        alignment = calculate_relationship_axis_alignment(
            part_a,
            part_b,
            candidate_axis
        )

        separation = calculate_relationship_separation(
            part_a,
            part_b,
            candidate_axis
        )

        weighted_alignment += (
            alignment * weight
        )

        weighted_separation += (
            separation * weight
        )

        total_weight += weight

        evidence_types = []

        for evidence in relationship.get(
            "evidence",
            []
        ):

            evidence_type = evidence.get(
                "type"
            )

            if evidence_type not in evidence_types:
                evidence_types.append(
                    evidence_type
                )

        pair_info = {
            "part_a": relationship["part_a"],
            "part_b": relationship["part_b"],
            "alignment": clean_number(alignment),
            "separation": clean_number(separation),
            "weight": clean_number(weight),
            "evidence": evidence_types
        }

        if alignment >= 0.5:
            supporting_pairs.append(
                pair_info
            )

        else:
            perpendicular_pairs.append(
                pair_info
            )

    if total_weight > 0:

        relationship_alignment = (
            weighted_alignment
            / total_weight
        )

        relationship_separation = (
            weighted_separation
            / total_weight
        )

    else:

        relationship_alignment = 0.0
        relationship_separation = 0.0

    same_layer_fraction = (
        calculate_same_layer_fraction(
            parts,
            relationships,
            candidate_axis
        )
    )

    same_layer_score = (
        1.0 - same_layer_fraction
    )

    spatial_distribution = (
        calculate_spatial_distribution_score(
            parts,
            candidate_axis
        )
    )

    # ------------------------------------------------------
    # Holistic score
    # ------------------------------------------------------
    #
    # Relationship alignment:
    #     45%
    #
    # Relationship separation:
    #     25%
    #
    # Layering:
    #     15%
    #
    # Overall spatial distribution:
    #     15%
    #
    # This intentionally prevents a simple "widest dimension"
    # heuristic from determining the assembly axis.
    # ------------------------------------------------------

    relationship_alignment_component = (
        relationship_alignment * 0.45
    )

    relationship_separation_component = (
        relationship_separation * 0.25
    )

    layering_component = (
        same_layer_score * 0.15
    )

    distribution_component = (
        spatial_distribution["score"] * 0.15
    )

    total_score = (
        relationship_alignment_component
        + relationship_separation_component
        + layering_component
        + distribution_component
    )

    return {
        "name": candidate_name,

        "vector": [
            clean_number(candidate_axis[0]),
            clean_number(candidate_axis[1]),
            clean_number(candidate_axis[2])
        ],

        "score": clean_number(
            total_score
        ),

        "evidence": {
            "relationship_alignment": {
                "score": clean_number(
                    relationship_alignment
                ),
                "weighted_score": clean_number(
                    weighted_alignment
                ),
                "total_relationship_weight":
                    clean_number(
                        total_weight
                    ),
                "supporting_pairs":
                    supporting_pairs,
                "perpendicular_pairs":
                    perpendicular_pairs
            },

            "relationship_separation": {
                "score": clean_number(
                    relationship_separation
                ),
                "weighted_score": clean_number(
                    weighted_separation
                )
            },

            "layering": {
                "score": clean_number(
                    same_layer_score
                ),
                "same_layer_fraction": clean_number(
                    same_layer_fraction
                )
            },

            "spatial_distribution": {
                "score": clean_number(
                    spatial_distribution["score"]
                ),
                "axis_extent": clean_number(
                    spatial_distribution["axis_extent"]
                ),
                "total_extent": clean_number(
                    spatial_distribution["total_extent"]
                )
            }
        }
    }


def get_axis_confidence(
    sorted_results
):
    """
    Estimate confidence from the separation between the best
    and second-best axis.

    The confidence is qualitative and only describes how
    decisively the algorithm separated the candidates.
    """

    if len(sorted_results) == 0:
        return "none"

    if len(sorted_results) == 1:
        return "high"

    best = sorted_results[0]["score"]
    second = sorted_results[1]["score"]

    difference = best - second

    if difference >= 0.25:
        return "high"

    if difference >= 0.10:
        return "medium"

    return "low"


def get_bbox_axis_projection_range(
    box,
    axis
):
    """
    Calculate the minimum and maximum projection of an AABB
    onto an arbitrary axis.

    All eight corners are considered.

    This is important because an occurrence's transform origin
    is not necessarily its physical bottom or top.
    """

    x_min = box["min"][0]
    y_min = box["min"][1]
    z_min = box["min"][2]

    x_max = box["max"][0]
    y_max = box["max"][1]
    z_max = box["max"][2]

    corners = [
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ]

    projections = []

    for corner in corners:

        projection = (
            corner[0] * axis[0]
            + corner[1] * axis[1]
            + corner[2] * axis[2]
        )

        projections.append(
            projection
        )

    return (
        min(projections),
        max(projections)
    )


def calculate_axis_base_score(
    part,
    parts,
    axis
):
    """
    Calculate a heuristic score for how likely a part is to be
    the foundational/base part along an already-selected axis.

    This is used only to orient the selected axis.

    Signals:

        - lower position along the axis
        - larger footprint perpendicular to the axis
        - number of connected relationships is handled separately

    The lowest part is the strongest signal.

    A larger perpendicular footprint provides supporting evidence
    because foundational parts commonly support parts above them.
    """

    min_projection, max_projection = (
        get_bbox_axis_projection_range(
            part["bounding_box"],
            axis
        )
    )

    # ------------------------------------------------------
    # Find global projection range.
    # ------------------------------------------------------

    all_min = float("inf")
    all_max = float("-inf")

    for other in parts:

        other_min, other_max = (
            get_bbox_axis_projection_range(
                other["bounding_box"],
                axis
            )
        )

        all_min = min(
            all_min,
            other_min
        )

        all_max = max(
            all_max,
            other_max
        )

    total_extent = all_max - all_min

    if total_extent < 0.000001:

        lower_score = 1.0

    else:

        # Lowest part gets 1.0.
        # Highest part gets 0.0.
        lower_score = (
            all_max - min_projection
        ) / total_extent

    # ------------------------------------------------------
    # Perpendicular footprint.
    # ------------------------------------------------------

    box = part["bounding_box"]

    axis_index = None

    if abs(axis[0]) > 0.999999:
        axis_index = 0

    elif abs(axis[1]) > 0.999999:
        axis_index = 1

    elif abs(axis[2]) > 0.999999:
        axis_index = 2

    footprint = 0.0

    if axis_index is not None:

        dimensions = [
            box["max"][0] - box["min"][0],
            box["max"][1] - box["min"][1],
            box["max"][2] - box["min"][2]
        ]

        perpendicular_dimensions = [
            dimensions[i]
            for i in range(3)
            if i != axis_index
        ]

        footprint = (
            perpendicular_dimensions[0]
            * perpendicular_dimensions[1]
        )

    return {
        "part": part["id"],
        "min_projection": clean_number(
            min_projection
        ),
        "max_projection": clean_number(
            max_projection
        ),
        "lower_score": clean_number(
            lower_score
        ),
        "footprint": clean_number(
            footprint
        )
    }


def choose_assembly_axis_direction(
    parts,
    selected_axis
):
    """
    Orient an already-selected UN-DIRECTED axis.

    Axis identification first determines:

        X vs Y vs Z

    This function then determines which end of that axis should
    represent the bottom of the assembly.

    For the current bottom-up assembly model, the foundational
    candidate is the part with the strongest combination of:

        - low position along the axis
        - large supporting footprint

    The axis is then oriented from that base toward the rest of
    the assembly.

    This keeps axis identification separate from assembly
    precedence.
    """

    axis = normalize_vector(
        selected_axis
    )

    if len(parts) == 0:
        return {
            "axis": axis,
            "name": None,
            "base_part": None,
            "candidates": []
        }

    base_candidates = []

    for part in parts:

        candidate = calculate_axis_base_score(
            part,
            parts,
            axis
        )

        base_candidates.append(
            candidate
        )

    # ------------------------------------------------------
    # Determine the base candidate.
    #
    # Lower position is the primary signal.
    # Footprint is a small tie-breaking/supporting signal.
    # ------------------------------------------------------

    base_candidates.sort(
        key=lambda candidate: (
            candidate["lower_score"],
            candidate["footprint"]
        ),
        reverse=True
    )

    base_part = base_candidates[0]

    base_min = base_part["min_projection"]

    # ------------------------------------------------------
    # Determine which sign points away from the base.
    #
    # We compare the base to the assembly's overall center.
    #
    # If the assembly extends toward +axis from the base,
    # retain +axis.
    #
    # If it extends toward -axis from the base, reverse it.
    # ------------------------------------------------------

    center_projection_sum = 0.0

    for part in parts:

        center = get_part_center(part)

        center_projection_sum += (
            center[0] * axis[0]
            + center[1] * axis[1]
            + center[2] * axis[2]
        )

    assembly_center_projection = (
        center_projection_sum
        / len(parts)
    )

    direction = list(axis)

    if assembly_center_projection < base_min:

        direction = [
            -axis[0],
            -axis[1],
            -axis[2]
        ]

    direction = normalize_vector(
        direction
    )

    # ------------------------------------------------------
    # Name
    # ------------------------------------------------------

    if vectors_are_close(
        direction,
        [1, 0, 0]
    ):
        name = "+X"

    elif vectors_are_close(
        direction,
        [-1, 0, 0]
    ):
        name = "-X"

    elif vectors_are_close(
        direction,
        [0, 1, 0]
    ):
        name = "+Y"

    elif vectors_are_close(
        direction,
        [0, -1, 0]
    ):
        name = "-Y"

    elif vectors_are_close(
        direction,
        [0, 0, 1]
    ):
        name = "+Z"

    elif vectors_are_close(
        direction,
        [0, 0, -1]
    ):
        name = "-Z"

    else:
        name = "custom"

    return {
        "axis": [
            clean_number(direction[0]),
            clean_number(direction[1]),
            clean_number(direction[2])
        ],

        "name": name,

        "base_part": base_part["part"],

        "candidates": base_candidates
    }


def analyze_assembly_axis(
    parts,
    relationships
):
    """
    Identify the global assembly axis using holistic CAD structure.

    IMPORTANT:

    This function identifies an AXIS LINE first.

        X
        Y
        Z

    It does not initially distinguish +X from -X, etc.

    The selected axis is then oriented separately so that the
    assembly proceeds from its inferred base toward higher layers.

    Motion/collision analysis is completely excluded.
    """

    # ------------------------------------------------------
    # Candidate AXIS LINES
    # ------------------------------------------------------

    candidates = [
        {
            "name": "X",
            "vector": [1, 0, 0]
        },
        {
            "name": "Y",
            "vector": [0, 1, 0]
        },
        {
            "name": "Z",
            "vector": [0, 0, 1]
        }
    ]

    results = []

    for candidate in candidates:

        result = calculate_assembly_axis_score(
            parts,
            relationships,
            candidate["name"],
            candidate["vector"]
        )

        results.append(
            result
        )

    # Highest score first.
    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    if len(results) == 0:

        return {
            "method":
                "holistic_geometry_relationship_analysis",

            "status":
                "no_candidates",

            "selected_axis":
                None,

            "selected_name":
                None,

            "confidence":
                "none",

            "direction_analysis":
                {},

            "candidates":
                []
        }

    best = results[0]

    relationship_weight = (
        best["evidence"][
            "relationship_alignment"
        ][
            "total_relationship_weight"
        ]
    )

    if relationship_weight <= 0:

        return {
            "method":
                "holistic_geometry_relationship_analysis",

            "status":
                "insufficient_relationship_evidence",

            "selected_axis":
                None,

            "selected_name":
                None,

            "confidence":
                "none",

            "direction_analysis":
                {},

            "candidates":
                results
        }

    confidence = get_axis_confidence(
        results
    )

    # ------------------------------------------------------
    # Separate axis-line selection from direction/sign.
    # ------------------------------------------------------

    direction_analysis = (
        choose_assembly_axis_direction(
            parts,
            best["vector"]
        )
    )

    return {
        "method":
            "holistic_geometry_relationship_analysis",

        "status":
            "complete",

        "selected_axis":
            direction_analysis["axis"],

        "selected_name":
            direction_analysis["name"],

        "axis_line":
            best["vector"],

        "axis_line_name":
            best["name"],

        "confidence":
            confidence,

        "direction_analysis":
            {
                "method":
                    "base_to_higher_layers",

                "base_part":
                    direction_analysis["base_part"],

                "selected_direction":
                    direction_analysis["axis"],

                "selected_direction_name":
                    direction_analysis["name"],

                "base_candidates":
                    direction_analysis["candidates"]
            },

        "candidates":
            results
    }


# ==========================================================
# CAD motion / collision testing
# ==========================================================

def make_translation_matrix(vector, distance):
    """Create a Fusion translation matrix for vector * distance."""

    transform = adsk.core.Matrix3D.create()

    translation = adsk.core.Vector3D.create(
        vector[0] * distance,
        vector[1] * distance,
        vector[2] * distance
    )

    transform.translation = translation

    return transform


def translate_bounding_box(box, direction, distance):
    """Translate an axis-aligned bounding box along a direction."""

    offset = [
        direction[0] * distance,
        direction[1] * distance,
        direction[2] * distance
    ]

    return {
        "min": [
            box["min"][0] + offset[0],
            box["min"][1] + offset[1],
            box["min"][2] + offset[2]
        ],
        "max": [
            box["max"][0] + offset[0],
            box["max"][1] + offset[1],
            box["max"][2] + offset[2]
        ]
    }


def aabb_overlaps(box_a, box_b, tolerance=0.0):
    """Return True when two AABBs overlap within a tolerance."""

    for axis in range(3):

        if box_a["max"][axis] < box_b["min"][axis] - tolerance:
            return False

        if box_b["max"][axis] < box_a["min"][axis] - tolerance:
            return False

    return True


def get_swept_aabb_interval(
    moving_box,
    blocker_box,
    direction,
    start_distance,
    end_distance
):
    """
    Calculate the distance interval during which two AABBs can
    overlap while the moving box travels along a direction.

    This is a broad-phase calculation.  It does not prove that the
    actual B-Rep solids collide; it only identifies distances worth
    checking with exact geometry.
    """

    interval_min = start_distance
    interval_max = end_distance

    for axis in range(3):

        velocity = direction[axis]
        moving_min = moving_box["min"][axis]
        moving_max = moving_box["max"][axis]
        blocker_min = blocker_box["min"][axis]
        blocker_max = blocker_box["max"][axis]

        if abs(velocity) < 0.000000001:

            if (
                moving_max < blocker_min
                or moving_min > blocker_max
            ):
                return None

            continue

        t1 = (
            blocker_min - moving_max
        ) / velocity

        t2 = (
            blocker_max - moving_min
        ) / velocity

        axis_min = min(t1, t2)
        axis_max = max(t1, t2)

        interval_min = max(
            interval_min,
            axis_min
        )

        interval_max = min(
            interval_max,
            axis_max
        )

        if interval_min > interval_max:
            return None

    return interval_min, interval_max


def get_occurrence_world_bodies(
    occurrence,
    temp_brep_manager
):
    """
    Create temporary world-coordinate copies of all solid B-Rep
    bodies belonging to an occurrence.

    The real Fusion model is never modified.
    """

    temporary_bodies = []

    component = occurrence.component
    bodies = component.bRepBodies
    occurrence_transform = occurrence.transform2

    for i in range(bodies.count):

        body = bodies.item(i)

        if not body:
            continue

        try:
            if body.volume <= 0:
                continue
        except Exception:
            continue

        copied_body = temp_brep_manager.copy(
            body
        )

        if not copied_body:
            continue

        if not temp_brep_manager.transform(
            copied_body,
            occurrence_transform
        ):
            continue

        temporary_bodies.append(
            copied_body
        )

    return temporary_bodies


def exact_body_collision(
    moving_bodies,
    blocker_bodies,
    temp_brep_manager,
    volume_tolerance=0.000000001
):
    """
    Perform exact B-Rep intersection tests between temporary bodies.

    Returns True only when the intersection has non-zero volume.
    Touching faces/edges therefore do not count as a collision.
    """

    intersection_type = (
        adsk.fusion.BooleanTypes.IntersectionBooleanType
    )

    for moving_body in moving_bodies:

        for blocker_body in blocker_bodies:

            moving_copy = temp_brep_manager.copy(
                moving_body
            )

            if not moving_copy:
                continue

            success = temp_brep_manager.booleanOperation(
                moving_copy,
                blocker_body,
                intersection_type
            )

            if not success:
                continue

            try:
                if moving_copy.volume > volume_tolerance:
                    return True
            except Exception:
                continue

    return False


def test_single_motion(
    moving_occurrence,
    blocker_occurrences,
    direction,
    max_distance,
    temp_brep_manager,
    epsilon=0.01
):
    """
    Test removal of one occurrence along one candidate direction.

    The occurrence is never moved in the real Fusion document.
    Temporary B-Rep copies are translated instead.
    """

    moving_bodies = get_occurrence_world_bodies(
        moving_occurrence,
        temp_brep_manager
    )

    if len(moving_bodies) == 0:
        return {
            "feasible": None,
            "status": "no_solid_geometry",
            "blocked_by": [],
            "distance": None
        }

    moving_box = get_bounding_box(
        moving_occurrence
    )

    blocker_data = []

    for blocker_occurrence in blocker_occurrences:

        blocker_bodies = get_occurrence_world_bodies(
            blocker_occurrence,
            temp_brep_manager
        )

        if len(blocker_bodies) == 0:
            continue

        blocker_data.append({
            "occurrence": blocker_occurrence,
            "bounding_box": get_bounding_box(
                blocker_occurrence
            ),
            "bodies": blocker_bodies
        })

    collision_candidates = []

    for blocker in blocker_data:

        interval = get_swept_aabb_interval(
            moving_box,
            blocker["bounding_box"],
            direction,
            epsilon,
            max_distance
        )

        if interval is None:
            continue

        interval_min, interval_max = interval

        if interval_max < epsilon:
            continue

        interval_min = max(
            interval_min,
            epsilon
        )

        if interval_min > max_distance:
            continue

        interval_max = min(
            interval_max,
            max_distance
        )

        if interval_min <= interval_max:

            collision_candidates.append({
                "blocker": blocker,
                "start": interval_min,
                "end": interval_max
            })

    collision_candidates.sort(
        key=lambda candidate: candidate["start"]
    )

    for candidate in collision_candidates:

        blocker = candidate["blocker"]
        start = candidate["start"]
        end = candidate["end"]

        test_distances = [start]

        if end > start:

            test_distances.append(
                (start + end) / 2.0
            )

            test_distances.append(
                end
            )

        for distance in test_distances:

            if (
                distance == start
                and end > start
            ):

                distance = min(
                    end,
                    start + 0.001
                )

            test_bodies = []

            for body in moving_bodies:

                copied_body = (
                    temp_brep_manager.copy(
                        body
                    )
                )

                if not copied_body:
                    continue

                transform = make_translation_matrix(
                    direction,
                    distance
                )

                if not temp_brep_manager.transform(
                    copied_body,
                    transform
                ):
                    continue

                test_bodies.append(
                    copied_body
                )

            if exact_body_collision(
                test_bodies,
                blocker["bodies"],
                temp_brep_manager
            ):

                return {
                    "feasible": False,
                    "status": "blocked",
                    "blocked_by": [
                        blocker["occurrence"].name
                    ],
                    "distance": clean_number(
                        distance
                    )
                }

    return {
        "feasible": True,
        "status": "collision_free",
        "blocked_by": [],
        "distance": clean_number(
            max_distance
        )
    }


def get_assembly_motion_distance(
    parts,
    multiplier=2.0
):
    """
    Choose a travel distance large enough to move a part clear of
    the whole assembly.
    """

    if len(parts) == 0:
        return 100.0

    min_point = [
        float("inf"),
        float("inf"),
        float("inf")
    ]

    max_point = [
        float("-inf"),
        float("-inf"),
        float("-inf")
    ]

    for part in parts:

        for axis in range(3):

            min_point[axis] = min(
                min_point[axis],
                part["bounding_box"]["min"][axis]
            )

            max_point[axis] = max(
                max_point[axis],
                part["bounding_box"]["max"][axis]
            )

    diagonal = vector_length([
        max_point[0] - min_point[0],
        max_point[1] - min_point[1],
        max_point[2] - min_point[2]
    ])

    return max(
        10.0,
        diagonal * multiplier
    )


def analyze_motion_directions(
    parts,
    fusion_occurrences,
    max_distance
):
    """
    Test every generated candidate direction against the CAD model.

    Motion analysis is independent from assembly-axis
    identification.
    """

    temp_brep_manager = (
        adsk.fusion.TemporaryBRepManager.get()
    )

    if not temp_brep_manager:
        return {
            "method": "temporary_brep_motion_test",
            "status": "unavailable",
            "results": []
        }

    occurrence_map = {}

    for occurrence in fusion_occurrences:
        occurrence_map[occurrence.name] = occurrence

    results = []

    for part in parts:

        occurrence = occurrence_map.get(
            part["id"]
        )

        if not occurrence:
            continue

        candidates = generate_candidate_directions(
            part["rotation"]
        )

        blocker_occurrences = [
            other
            for other in fusion_occurrences
            if other.name != occurrence.name
        ]

        direction_results = []

        for candidate in candidates:

            motion = test_single_motion(
                occurrence,
                blocker_occurrences,
                candidate["vector"],
                max_distance,
                temp_brep_manager
            )

            result = {
                "name": candidate["name"],
                "vector": candidate["vector"],
                "frame": candidate["frame"],
                "source": candidate["source"],
                "removal": motion
            }

            if motion.get("feasible") is True:

                result["insertion"] = {
                    "vector": [
                        clean_number(
                            -candidate["vector"][0]
                        ),
                        clean_number(
                            -candidate["vector"][1]
                        ),
                        clean_number(
                            -candidate["vector"][2]
                        )
                    ],
                    "derived_from":
                        "reverse_of_collision_free_removal"
                }

            direction_results.append(
                result
            )

        results.append({
            "part": part["id"],
            "max_travel_distance":
                clean_number(
                    max_distance
                ),
            "directions":
                direction_results
        })

    return {
        "method":
            "temporary_brep_motion_test",

        "status":
            "complete",

        "max_travel_distance":
            clean_number(
                max_distance
            ),

        "results":
            results
    }


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
            graph[part_a].append(
                part_b
            )

        if part_a not in graph[part_b]:
            graph[part_b].append(
                part_a
            )

    return graph


def get_part_height_map(
    parts,
    assembly_axis
):
    """
    Calculate each part's scalar bottom position along the
    selected assembly axis.

    IMPORTANT:

    This uses the minimum projection of the part's entire
    bounding box onto the assembly axis.

    It does NOT use:

        occurrence.position

    because the occurrence origin is not necessarily the
    physical bottom of the part.

    For example, if:

        External1 bbox = Z 3.999 -> 9.001
        External3 bbox = Z 8.999 -> 17.001

    then their assembly heights are:

        External1 = 3.999
        External3 = 8.999

    for +Z assembly.
    """

    axis = normalize_vector(
        assembly_axis
    )

    heights = {}

    for part in parts:

        min_projection, max_projection = (
            get_bbox_axis_projection_range(
                part["bounding_box"],
                axis
            )
        )

        heights[part["id"]] = clean_number(
            min_projection
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

    Important distinction:

        joint graph       = which parts are connected
        precedence graph  = which connected part comes first

    For the current bottom-up assembly model:

        lower part -> higher part

    Equal-height connected parts remain ambiguous.

    The height is the minimum bounding-box projection along the
    selected assembly axis.
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

            pair_key = tuple(sorted([
                part_a,
                part_b
            ]))

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

            precedence_graph[
                before
            ].append(
                after
            )

    # ------------------------------------------------------
    # Roots and leaves.
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
    # Connected components.
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
            "fusion_joint_graph_plus_assembly_axis_bbox_height",

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

        # Assembly axis is determined automatically after
        # relationships have been extracted.
        assembly_axis = None

        assembly = {
            "assembly": {
                "name": root.name,
                "units": "mm",
                "assembly_axis": None
            },

            "components": [],

            "parts": [],

            "relationships": [],

            "direction_analysis": [],

            "axis_analysis": {},

            "precedence_analysis": {},

            "motion_analysis": {}
        }

        # ==================================================
        # Track unique component definitions
        # ==================================================

        components_seen = set()

        # ==================================================
        # Store occurrence information
        # ==================================================

        occurrences = []

        # Keep actual Fusion occurrence objects separately
        # for temporary-B-Rep motion testing.
        fusion_occurrences = []

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

            component_id = (
                occurrence.component.name
            )

            # ----------------------------------------------
            # Component definition
            # ----------------------------------------------

            if component_id not in components_seen:

                component = occurrence.component

                assembly["components"].append({
                    "id":
                        component_id,

                    "bounding_box":
                        get_component_bounds(
                            component
                        )
                })

                components_seen.add(
                    component_id
                )

            # ----------------------------------------------
            # Retain raw +Z height for exported information.
            #
            # Precedence does NOT use this value anymore.
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

                "rotation":
                    rotation,

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
            # Relationship-analysis representation
            # ----------------------------------------------

            occurrences.append({
                "id":
                    occurrence.name,

                "bounding_box":
                    bounding_box
            })

            fusion_occurrences.append(
                occurrence
            )

        # ==================================================
        # Geometric relationships
        # ==================================================

        geometric_relationships = (
            find_geometric_relationships(
                occurrences,
                tolerance=0.1
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
        # Candidate directions
        #
        # These are still used only for individual-part
        # motion analysis.
        # ==================================================

        assembly["direction_analysis"] = (
            analyze_candidate_directions(
                assembly["parts"]
            )
        )

        # ==================================================
        # GLOBAL ASSEMBLY-AXIS IDENTIFICATION
        # ==================================================
        #
        # Stage 1:
        #
        #     Identify X vs Y vs Z.
        #
        # Stage 2:
        #
        #     Orient the selected axis into an assembly
        #     direction.
        #
        # Motion/collision is NOT involved.
        # ==================================================

        assembly["axis_analysis"] = (
            analyze_assembly_axis(
                assembly["parts"],
                assembly["relationships"]
            )
        )

        selected_axis = (
            assembly["axis_analysis"].get(
                "selected_axis"
            )
        )

        if selected_axis is None:

            ui.messageBox(
                "Assembly-axis identification could not "
                "determine an axis from the available "
                "CAD relationships."
            )

            return

        assembly_axis = selected_axis

        assembly["assembly"][
            "assembly_axis"
        ] = assembly_axis

        # ==================================================
        # CAD motion / collision analysis
        # ==================================================
        #
        # This remains completely independent of the global
        # assembly-axis decision.
        # ==================================================

        max_motion_distance = (
            get_assembly_motion_distance(
                assembly["parts"]
            )
        )

        assembly["motion_analysis"] = (
            analyze_motion_directions(
                assembly["parts"],
                fusion_occurrences,
                max_motion_distance
            )
        )

        # ==================================================
        # Precedence analysis
        #
        # This happens AFTER assembly-axis identification.
        #
        # Heights are derived from the bounding-box projection
        # onto the selected assembly axis.
        # ==================================================

        assembly["precedence_analysis"] = (
            analyze_joint_precedence(
                assembly["parts"],
                assembly["relationships"],
                assembly_axis
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
        # Confirmation
        # ==================================================

        axis_name = (
            assembly["axis_analysis"].get(
                "selected_name",
                "unknown"
            )
        )

        axis_confidence = (
            assembly["axis_analysis"].get(
                "confidence",
                "unknown"
            )
        )

        precedence_edges = (
            assembly[
                "precedence_analysis"
            ][
                "precedence_edges"
            ]
        )

        ui.messageBox(
            "Export complete!\n\n"
            + "Parts: "
            + str(
                len(
                    assembly["parts"]
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
            + "Assembly axis: "
            + axis_name
            + "\n"
            + "Axis confidence: "
            + axis_confidence
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
            + "Motion analyses: "
            + str(
                len(
                    assembly[
                        "motion_analysis"
                    ].get(
                        "results",
                        []
                    )
                )
            )
            + "\n"
            + "Precedence edges: "
            + str(
                len(
                    precedence_edges
                )
            )
            + "\n\n"
            + "Saved to:\n"
            + file_path
        )

    except Exception as e:

        ui.messageBox(
            "ERROR:\n\n"
            + str(e)
        )