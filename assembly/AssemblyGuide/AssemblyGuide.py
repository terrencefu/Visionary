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

def get_component_appearance(component):
    """
    Get a representative appearance/color for a component.

    Fusion exposes the current appearance on each B-Rep body.
    For the Visionary parts we normally expect one body/one
    appearance, but this function handles multiple bodies by
    returning all distinct body appearances it can read.

    Returns:
        {
            "name": <appearance name or None>,
            "rgb": [r, g, b] or None,
            "hex": "#RRGGBB" or None,
            "appearances": [...]
        }
        or None if no readable appearance is available.
    """

    appearances = []
    seen = set()

    for body in component.bRepBodies:

        try:
            appearance = body.appearance
        except Exception:
            appearance = None

        if not appearance:
            continue

        try:
            color = appearance.color
        except Exception:
            color = None

        if not color:
            continue

        try:
            rgb = [
                int(color.red),
                int(color.green),
                int(color.blue)
            ]
        except Exception:
            continue

        try:
            name = appearance.name
        except Exception:
            name = None

        key = (name, tuple(rgb))

        if key in seen:
            continue

        seen.add(key)

        appearances.append({
            "name": name,
            "rgb": rgb,
            "hex": "#{:02X}{:02X}{:02X}".format(
                rgb[0],
                rgb[1],
                rgb[2]
            )
        })

    if not appearances:
        return None

    # Keep the simple top-level fields convenient for the CV pipeline.
    # If a component has multiple distinct body appearances, the first
    # one is treated as the representative appearance while the complete
    # list is preserved in "appearances".
    representative = appearances[0]

    return {
        "name": representative["name"],
        "rgb": representative["rgb"],
        "hex": representative["hex"],
        "appearances": appearances
    }


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

    This stage generates candidate directions only. It does
    not yet determine whether a direction is geometrically
    feasible.
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

        # The moving interval overlaps the blocker interval when:
        #
        #   moving_min + v*t <= blocker_max
        #   moving_max + v*t >= blocker_min
        #
        # Solve both inequalities for t.

        t1 = (blocker_min - moving_max) / velocity
        t2 = (blocker_max - moving_min) / velocity

        axis_min = min(t1, t2)
        axis_max = max(t1, t2)

        interval_min = max(interval_min, axis_min)
        interval_max = min(interval_max, axis_max)

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

        # Motion/collision testing currently targets solid bodies.
        # Wires and surfaces have zero volume and are ignored.
        try:
            if body.volume <= 0:
                continue
        except Exception:
            continue

        copied_body = temp_brep_manager.copy(body)

        if not copied_body:
            continue

        if not temp_brep_manager.transform(
            copied_body,
            occurrence_transform
        ):
            continue

        temporary_bodies.append(copied_body)

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
                # A Boolean failure is not automatically treated as
                # a collision.  The broad-phase already told us this
                # pair is geometrically close, but the kernel may not
                # be able to resolve a tangency or degenerate case.
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

    Returns a result describing whether the motion is collision-free
    or blocked, including the first blocker that produced an exact
    B-Rep collision.
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

    # Create temporary world-coordinate blocker bodies once.  They
    # remain stationary while the moving copies are translated.
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

    # ------------------------------------------------------
    # Build broad-phase collision intervals.
    # ------------------------------------------------------

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

        interval_min = max(interval_min, epsilon)

        if interval_min > max_distance:
            continue

        interval_max = min(interval_max, max_distance)

        if interval_min <= interval_max:
            collision_candidates.append({
                "blocker": blocker,
                "start": interval_min,
                "end": interval_max
            })

    # Check the earliest broad-phase intervals first.
    collision_candidates.sort(
        key=lambda candidate: candidate["start"]
    )

    # ------------------------------------------------------
    # Exact B-Rep checks.
    # ------------------------------------------------------
    # AABB overlap is deliberately only a broad phase.  We check a
    # few points inside each interval because the exact collision
    # may occupy only part of the broad-phase interval.

    for candidate in collision_candidates:

        blocker = candidate["blocker"]
        start = candidate["start"]
        end = candidate["end"]

        test_distances = [start]

        if end > start:
            test_distances.append(
                (start + end) / 2.0
            )
            test_distances.append(end)

        for distance in test_distances:

            # Avoid testing exactly at a pure contact boundary when
            # possible.  The initial epsilon serves the same purpose
            # for the first test.
            if distance == start and end > start:
                distance = min(
                    end,
                    start + 0.001
                )

            test_bodies = []

            for body in moving_bodies:

                copied_body = temp_brep_manager.copy(
                    body
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

                test_bodies.append(copied_body)

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
                    "distance": clean_number(distance)
                }

    # No exact collision was found anywhere along the broad-phase
    # intervals, so this direction is considered collision-free.
    return {
        "feasible": True,
        "status": "collision_free",
        "blocked_by": [],
        "distance": clean_number(max_distance)
    }


def get_assembly_motion_distance(parts, multiplier=2.0):
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

    The output distinguishes:

        collision_free
        blocked
        no_solid_geometry

    For a collision-free removal direction, the corresponding
    insertion direction is simply the opposite vector.
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
                        clean_number(-candidate["vector"][0]),
                        clean_number(-candidate["vector"][1]),
                        clean_number(-candidate["vector"][2])
                    ],
                    "derived_from": "reverse_of_collision_free_removal"
                }

            direction_results.append(result)

        results.append({
            "part": part["id"],
            "max_travel_distance": clean_number(max_distance),
            "directions": direction_results
        })

    return {
        "method": "temporary_brep_motion_test",
        "status": "complete",
        "max_travel_distance": clean_number(max_distance),
        "results": results
    }


# ==========================================================
# Joint graph and precedence analysis
# ==========================================================

def get_joint_graph(relationships):
    """
    Build an undirected graph from high-confidence Fusion joint
    relationships.

    The graph represents physical/CAD connectivity, not assembly
    order.  Direction is deliberately added later by the precedence
    analysis using the assembly axis and part heights.
    """

    graph = {}

    for relationship in relationships:

        has_joint_evidence = False

        for evidence in relationship.get("evidence", []):

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


def get_part_height_map(parts, assembly_axis):
    """
    Calculate each part's scalar position along the assembly axis.

    For the current prototype, assembly_height is already based on
    the +Z bounding-box minimum.  For a general axis, this function
    falls back to the occurrence position projected onto that axis.
    """

    axis = normalize_vector(assembly_axis)
    heights = {}

    for part in parts:

        if vectors_are_close(axis, [0, 0, 1]):
            height = part.get("assembly_height", None)

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

        heights[part["id"]] = clean_number(height)

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
        "height_before": clean_number(height_before),
        "height_after": clean_number(height_after)
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

    A Fusion joint by itself is not an ordering statement.  For the
    current Visionary bottom-up assembly model, a joint connecting a
    lower part to a higher part gives the precedence relation:

        lower part -> higher part

    Equal-height pairs are kept as ambiguous rather than being given
    an arbitrary order.

    This function intentionally produces a partial order.  It does
    not force a single assembly sequence when the CAD model permits
    branches.
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

            pair_key = tuple(sorted([part_a, part_b]))

            if pair_key in processed_pairs:
                continue

            processed_pairs.add(pair_key)

            height_a = heights.get(part_a, 0.0)
            height_b = heights.get(part_b, 0.0)

            height_difference = height_b - height_a

            if abs(height_difference) <= height_tolerance:

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
            precedence_graph[before].append(after)

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

        if len(precedence_graph[part]) == 0:
            leaves.append(part)

    # Sort roots/leaves to make JSON deterministic.
    roots.sort()
    leaves.sort()

    # ------------------------------------------------------
    # Find connected components of the joint graph.
    # ------------------------------------------------------

    connected_components = []
    visited = set()

    for start in sorted(joint_graph.keys()):

        if start in visited:
            continue

        component = []
        stack = [start]
        visited.add(start)

        while stack:

            current = stack.pop()
            component.append(current)

            for neighbor in joint_graph.get(current, []):

                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        component.sort()
        connected_components.append(component)

    return {
        "method": "fusion_joint_graph_plus_assembly_axis_height",
        "assembly_axis": assembly_axis,
        "joint_graph": joint_graph,
        "part_heights": heights,
        "precedence_graph": precedence_graph,
        "precedence_edges": precedence_edges,
        "ambiguous_pairs": ambiguous_pairs,
        "roots": roots,
        "leaves": leaves,
        "connected_components": connected_components
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

        # Current Visionary prototype uses +Z as the primary
        # assembly axis.
        assembly_axis = [0, 0, 1]

        assembly = {
            "assembly": {
                "name": root.name,
                "units": "mm",
                "assembly_axis": assembly_axis
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

        # Keep the actual Fusion occurrence objects separately for
        # non-destructive temporary-B-Rep motion testing.
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
                        ),
                    "appearance":
                        get_component_appearance(
                            component
                        )
                })

                components_seen.add(
                    component_id
                )

            # ----------------------------------------------
            # Assembly height
            #
            # Current assembly axis is +Z.
            #
            # Use the bottom of the occurrence bounding box
            # rather than the occurrence origin.
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

            fusion_occurrences.append(occurrence)

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
        # Candidate direction generation
        # ==================================================

        assembly["direction_analysis"] = (
            analyze_candidate_directions(
                assembly["parts"]
            )
        )

        # ==================================================
        # CAD motion / collision analysis
        # ==================================================

        max_motion_distance = get_assembly_motion_distance(
            assembly["parts"]
        )

        assembly["motion_analysis"] = (
            analyze_motion_directions(
                assembly["parts"],
                fusion_occurrences,
                max_motion_distance
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

            f.write(json_text)

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
            + "Direction analyses: "
            + str(
                len(assembly["direction_analysis"])
            )
            + "\n"
            + "Motion analyses: "
            + str(
                len(assembly["motion_analysis"].get("results", []))
            )
            + "\n"
            + "Precedence edges: "
            + str(
                len(assembly["precedence_analysis"]["precedence_edges"])
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
