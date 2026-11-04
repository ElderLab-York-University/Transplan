"""Regression tests for reloading homography CSVs and high-res outputs.

No Qt/display server is required for these noninteractive tests.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'homographygui'))
from correspondences import load_correspondences, save_correspondences
from backend import backend
from homography_visualization import save_homography_visualizations


def test_csv_roundtrip_unlimited_and_editable(tmp_path):
    csv_path = tmp_path / 'test.homography.EmeryOrtho.csv'
    camera = [[float(i * 7 + 0.25), float(i * 13)] for i in range(49)]
    top = [[float(i * 9), float(i * 10 + 0.75)] for i in range(49)]
    save_correspondences(str(csv_path), camera, top)
    a, b = load_correspondences(str(csv_path))
    assert len(a) == len(b) == 49
    assert a == camera and b == top
    a[12] = [33.5, 45.5]
    save_correspondences(str(csv_path), a, b)
    assert load_correspondences(str(csv_path))[0][12] == [33.5, 45.5]


def test_csv_empty_missing_and_errors(tmp_path):
    assert load_correspondences(str(tmp_path / 'missing.csv')) == ([], [])
    csv_path = tmp_path / 'bad.csv'
    csv_path.write_text('1,2,3\n')
    with pytest.raises(ValueError, match='columns'):
        load_correspondences(str(csv_path))
    csv_path.write_text('1,nan,3,4\n')
    with pytest.raises(ValueError, match='NaN'):
        load_correspondences(str(csv_path))
    with pytest.raises(ValueError, match='point'):
        save_correspondences(str(csv_path), [[1, 2]], [])


def test_backend_writes_legacy_compatible_matrix(tmp_path):
    street_path = tmp_path / 'street.png'
    top_path = tmp_path / 'top.png'
    cv2.imwrite(str(street_path), np.full((80, 120, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(top_path), np.full((100, 160, 3), 100, dtype=np.uint8))
    b = backend()
    b.setFileName1(str(street_path))
    b.setFileName2(str(top_path))
    b.set_npy_file_path(str(tmp_path / 'result.npy'))
    b.set_txt_file_path(str(tmp_path / 'result.txt'))
    points1 = [[0, 0], [90, 0], [90, 60], [0, 60], [45, 20]]
    points2 = [[10, 15], [100, 15], [100, 75], [10, 75], [55, 35]]
    image = b.doHomography(points1, points2)
    assert image.shape[:2] == (100, 160)
    stored = np.load(tmp_path / 'result.npy', allow_pickle=True)
    assert stored[0].shape == (3, 3)
    assert np.allclose(stored[0].astype(float), np.loadtxt(tmp_path / 'result.txt', delimiter=','))
    assert np.allclose(stored[0].astype(float), [[1, 0, 10], [0, 1, 15], [0, 0, 1]], atol=1e-5)


def test_full_res_overlay_and_dpi(tmp_path):
    street = np.zeros((180, 240, 3), dtype=np.uint8)
    street[:, 0:120] = (0, 100, 255)
    top = np.full((260, 340, 3), 190, dtype=np.uint8)
    street_path = tmp_path / 'street.png'
    top_path = tmp_path / 'top.png'
    cv2.imwrite(str(street_path), street)
    cv2.imwrite(str(top_path), top)
    matrix = np.array([[1, 0, 13], [0, 1, 27], [0, 0, 1]], dtype=float)
    figure_100, raw_100 = save_homography_visualizations(
        str(street_path), str(top_path), matrix, str(tmp_path / '100.png'), dpi=100
    )
    figure_200, raw_200 = save_homography_visualizations(
        str(street_path), str(top_path), matrix, str(tmp_path / '200.png'), dpi=200
    )
    overlay = cv2.imread(raw_200)
    assert overlay.shape[:2] == top.shape[:2]
    assert np.array_equal(cv2.imread(raw_100), overlay)
    assert (overlay[0, 0] == top[0, 0]).all()  # outside coverage remains unchanged
    assert cv2.imread(figure_200).shape[1] > cv2.imread(figure_100).shape[1] * 1.8
