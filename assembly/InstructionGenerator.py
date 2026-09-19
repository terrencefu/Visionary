import json


def load_assembly(filename):
    with open(filename, "r") as f:
        return json.load(f)


def find_part(assembly, part_id):
    for part in assembly["parts"]:
        if part["id"] == part_id:
            return part

    return None


def create_step(step_number, part):
    return {
        "id": f"step_{step_number}",
        "part": part["id"],
        "action": "place",
        "instruction": f"Place {part['component']}.",
        "target": {
            "position": part["position"],
            "rotation": part["rotation"]
        },
        "verification": {
            "position_tolerance_mm": 1.0,
            "rotation_tolerance_deg": 5.0
        }
    }


def generate_instructions(assembly, sequence):
    steps = []

    for step_number, part_id in enumerate(sequence, start=1):

        part = find_part(assembly, part_id)

        if part is None:
            print(f"WARNING: Could not find {part_id}")
            continue

        step = create_step(step_number, part)
        steps.append(step)

    return {
        "assembly": assembly["assembly"],
        "steps": steps
    }


def save_instructions(instructions, filename):
    with open(filename, "w") as f:
        json.dump(instructions, f, indent=4)


def main():

    # Load Fusion's exported CAD data
    assembly = load_assembly("assembly.json")

    # TEMPORARY:
    # We will replace this with automatic sequencing later.
    sequence = [
        "External1:1",
        "External4:1",
        "External3:1"
    ]

    instructions = generate_instructions(
        assembly,
        sequence
    )

    save_instructions(
        instructions,
        "instructions.json"
    )

    print("Generated instructions.json")


if __name__ == "__main__":
    main()