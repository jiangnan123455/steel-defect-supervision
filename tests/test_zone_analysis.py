"""九宫格区域分析测试（项目核心创新点）。"""

import numpy as np
import pytest
import supervision as sv

from src.zone_analysis import (
    ZONE_NAMES,
    analyze_zones,
    build_grid_polygons,
)


def _detections(xyxy, class_id=None):
    xyxy = np.array(xyxy, dtype=np.float32)
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.linspace(0.9, 0.5, n, dtype=np.float32),
        class_id=np.zeros(n, dtype=int) if class_id is None else np.array(class_id),
    )


def test_zone_names_cover_nine_cells_in_reading_order():
    assert len(ZONE_NAMES) == 9
    assert ZONE_NAMES[0] == "左上"
    assert ZONE_NAMES[4] == "中央"
    assert ZONE_NAMES[8] == "右下"


def test_build_grid_polygons_returns_nine_quads_covering_full_image():
    polygons = build_grid_polygons(300, 600)

    assert len(polygons) == 9
    for polygon in polygons:
        assert polygon.shape == (4, 2)
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        assert x_min >= 0 and y_min >= 0
        assert x_max <= 300 and y_max <= 600

    # 九宫格应覆盖整幅图：所有多边形的坐标并集达到图像尺寸
    all_points = np.vstack(polygons)
    assert all_points[:, 0].max() == 300
    assert all_points[:, 1].max() == 600


def test_detection_center_in_top_left_cell_is_counted_top_left():
    detections = _detections([[10, 10, 50, 50]])

    result = analyze_zones(detections, width=300, height=300)

    assert result.counts["左上"] == 1
    assert sum(result.counts.values()) == 1


def test_detection_center_in_bottom_right_cell_is_counted_bottom_right():
    detections = _detections([[250, 250, 290, 290]])

    result = analyze_zones(detections, width=300, height=300)

    assert result.counts["右下"] == 1


def test_nine_cell_mapping_covers_all_positions():
    boxes = []
    expected = []
    centers = [
        (50, 50), (150, 50), (250, 50),
        (50, 150), (150, 150), (250, 150),
        (50, 250), (150, 250), (250, 250),
    ]
    for cx, cy in centers:
        boxes.append([cx - 10, cy - 10, cx + 10, cy + 10])
    names = [
        "左上", "中上", "右上",
        "左中", "中央", "右中",
        "左下", "中下", "右下",
    ]
    for name in names:
        expected.append(name)

    result = analyze_zones(_detections(boxes), width=300, height=300)

    for name in expected:
        assert result.counts[name] == 1, name
    assert result.hotspot is None or result.hotspot_count >= 1


def test_zone_counts_sum_equals_detection_count():
    detections = _detections([
        [10, 10, 50, 50],
        [140, 140, 160, 160],
        [240, 240, 290, 290],
    ])

    result = analyze_zones(detections, width=300, height=300)

    assert sum(result.counts.values()) == 3
    assert result.assigned_count == 3
    assert result.unassigned_count == 0


def test_detection_on_exact_cell_boundary_is_assigned_to_exactly_one_cell():
    # 中心点正好落在 x=100 与 y=100 的分割线上
    detections = _detections([[90, 90, 110, 110]])

    result = analyze_zones(detections, width=300, height=300)

    assert sum(result.counts.values()) == 1
    assert result.assigned_count == 1


def test_hotspot_reports_cell_with_most_defects():
    detections = _detections([
        [10, 10, 40, 40],
        [50, 50, 80, 80],
        [110, 10, 140, 40],
    ])

    result = analyze_zones(detections, width=300, height=300)

    assert result.hotspot == "左上"
    assert result.hotspot_count == 2


def test_empty_detections_gives_zero_counts():
    result = analyze_zones(sv.Detections.empty(), width=300, height=300)

    assert sum(result.counts.values()) == 0
    assert result.hotspot is None
    assert result.hotspot_count == 0


def test_class_breakdown_is_reported_per_zone():
    detections = _detections(
        [[10, 10, 50, 50], [60, 60, 90, 90]],
        class_id=[0, 2],
    )

    result = analyze_zones(detections, width=300, height=300)

    assert result.class_counts["左上"] == {0: 1, 2: 1}


def test_detection_entirely_outside_image_is_unassigned():
    detections = _detections([[400, 400, 450, 450]])

    result = analyze_zones(detections, width=300, height=300)

    assert result.assigned_count == 0
    assert result.unassigned_count == 1
    assert sum(result.counts.values()) == 0


def test_zero_sized_image_is_rejected():
    with pytest.raises(ValueError):
        analyze_zones(sv.Detections.empty(), width=0, height=300)
