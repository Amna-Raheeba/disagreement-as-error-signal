"""Regenerate the seeded dataset summary used by the README.

The dataset itself is generated in code (disagreement_signal.data); this
script prints the split sizes and region counts so they can be checked
against the documentation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disagreement_signal.data import region_counts, split_dataset


def main():
    train, val, test = split_dataset(seed=7)
    for name, part in (("train", train), ("val", val), ("test", test)):
        rows, labels, regions = part
        print(
            "%s: %d rows, %d positive, regions %s"
            % (name, len(rows), sum(labels), region_counts(regions))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
