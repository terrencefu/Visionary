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

        # Normalize overlap by the smaller part dimension.
        #
        # This is deliberately different from center-to-center
        # distance.
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

    # A stacking/interface axis should separate the two parts
    # while retaining substantial support/alignment in the other
    # two axes.
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

        # A pair with both joint and geometric evidence is still
        # one high-confidence relationship, not two independent
        # relationships.
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

    This separation is important.

    For example, a long 1x8 plate can have a center many
    centimeters away from the engine it supports, even though
    their actual interface is vertically stacked.

    The old center-displacement method could therefore select
    Y instead of Z.
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

    # ------------------------------------------------------
    # Score each relationship independently.
    # ------------------------------------------------------

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

        # --------------------------------------------------
        # Center displacement is retained ONLY as diagnostic
        # information.
        #
        # It does not contribute to axis_totals.
        # --------------------------------------------------

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

    # ------------------------------------------------------
    # Normalize axis scores.
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Select the axis line.
    # ------------------------------------------------------

    high_confidence_relationship_count = sum(
        1
        for relationship in relationship_diagnostics
        if relationship["weight"] >= 2.0
    )

    if max(axis_weights) <= 0.0:

        # No relationship information exists.
        #
        # Keep the prototype's conventional +Z fallback,
        # but explicitly mark it low confidence.
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
    # Determine the sign separately.
    # ------------------------------------------------------
    #
    # The axis line above is deliberately X/Y/Z without a
    # sign.
    #
    # Centers are used here ONLY to determine whether the
    # selected axis should be + or -.
    #
    # They are NOT used to decide which axis wins.
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

    # Check for duplicate directions.

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

    Global directions:
        +/- X
        +/- Y
        +/- Z

    Local directions:
        +/- x
        +/- y
        +/- z

    The local axes are obtained from the columns of
    the part's 3x3 rotation matrix.

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
    order. Direction is deliberately added later by the precedence
    analysis using the assembly axis and part heights.
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

    Important distinction:

        joint graph       = which parts are connected
        precedence graph  = which connected part must come first

    A Fusion joint by itself is not an ordering statement.

    For the current Visionary bottom-up assembly model, a joint
    connecting a lower part to a higher part gives:

        lower part -> higher part

    Equal-height pairs are kept as ambiguous rather than being
    given an arbitrary order.

    This function intentionally produces a partial order.
    It does not force a single assembly sequence when the CAD
    model permits branches.
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
    # Build directed adjacency from the precedence edges.
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
    # Identify graph roots and leaves.
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

    # Sort roots/leaves to make JSON deterministic.

    roots.sort()
    leaves.sort()

    # ------------------------------------------------------
    # Find connected components of the joint graph.
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

            "precedence_analysis": {}
        }

        # ==================================================
        # Track unique component definitions
        # ==================================================

        components_seen = set()

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

                assembly["components"].append({
                    "id": component_id,

                    "bounding_box":
                        get_component_bounds(
                            component
                        )
                })

                components_seen.add(
                    component_id
                )

            # ----------------------------------------------
            # Assembly height
            #
            # This remains the bottom of the occurrence
            # bounding box along +Z.
            #
            # It is used by precedence AFTER the global
            # assembly axis has been selected.
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
        # Assembly axis identification
        # ==================================================
        #
        # IMPORTANT:
        #
        # This is now determined from relationship/interface
        # geometry rather than center-to-center displacement.
        #
        # Motion/collision analysis is deliberately NOT part
        # of this decision.
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
        # Precedence analysis from Fusion joint graph
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
            + "\n\n"
            + "Saved to:\n"
            + file_path
        )

    except Exception as e:

        ui.messageBox(
            "ERROR:\n\n"
            + str(e)
        )