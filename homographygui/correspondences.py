"""Read/write Transplan's original four-column homography correspondence CSVs.

The coordinates are stored in the *original image pixel coordinates*, not
coordinates of the scaled GUI viewport.
"""
import csv
import math
import os


def load_correspondences(path):
    """Return (street_points, top_points), with no limit on the row count."""
    if not os.path.isfile(path):
        return [], []

    street, top = [], []
    with open(path, newline="") as f:
        for line, row in enumerate(csv.reader(f), start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) != 4:
                raise ValueError("Correspondence CSV line {} has {} columns, expected 4".format(line, len(row)))
            try:
                coords = [float(value) for value in row]
            except ValueError:
                raise ValueError("Correspondence CSV line {} contains nonnumeric values".format(line))
            if not all(math.isfinite(value) for value in coords):
                raise ValueError("Correspondence CSV line {} contains NaN/infinite values".format(line))
            street.append(coords[:2])
            top.append(coords[2:])
    return street, top


def save_correspondences(path, street, top):
    """Write the same four-column format; reject incomplete pairs."""
    if len(street) != len(top):
        raise ValueError("Each street-view point must have one top-view point")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for a, b in zip(street, top):
            writer.writerow([a[0], a[1], b[0], b[1]])
