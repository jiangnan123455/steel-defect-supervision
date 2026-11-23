"""九宫格缺陷空间分布分析。

本模块是项目的核心创新点：把检测结果从"类别 + 边界框"扩展为
"缺陷在产品表面九个区域中的分布情况"。

实现方式遵循方案书要求：使用 Supervision 的 `sv.PolygonZone` 为每个区域
构建多边形区域，并以检测框中心点作为统一锚点判断目标归属。

边界归属规则
------------
九个区域互不重叠，但相邻区域共享分割线。为保证"区域计数之和等于参与统计
的检测目标数"这一验收标准（T08），实现上：

1. 先用 `PolygonZone.trigger` 判断每个检测框中心点落在哪些多边形内；
2. 若一个中心点同时命中多个区域（只可能发生在分割线上），按固定顺序
   （先上后下、先左后右）取第一个命中区域，保证归属唯一且可复现；
3. 中心点落在图像范围之外的检测框记为"未归属"，不计入九宫格计数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

ZONE_NAMES: tuple[str, ...] = (
    "左上",
    "中上",
    "右上",
    "左中",
    "中央",
    "右中",
    "左下",
    "中下",
    "右下",
)

GRID_ROWS = 3
GRID_COLS = 3


@dataclass
class ZoneAnalysis:
    """九宫格区域统计结果。"""

    counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, dict[int, int]] = field(default_factory=dict)
    assigned_count: int = 0
    unassigned_count: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def hotspot(self) -> str | None:
        """缺陷数量最多的区域；没有任何缺陷时返回 None。"""
        if not self.counts or self.total == 0:
            return None
        return max(self.counts.items(), key=lambda item: (item[1], -ZONE_NAMES.index(item[0])))[0]

    @property
    def hotspot_count(self) -> int:
        if self.hotspot is None:
            return 0
        return self.counts[self.hotspot]


def build_grid_polygons(
    width: int,
    height: int,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> list[np.ndarray]:
    """把图像平均切成 rows × cols 个矩形区域，返回多边形顶点数组。

    每个多边形为形状 `(4, 2)` 的整数数组，顶点顺序为左上、右上、右下、左下。
    列表顺序按行优先排列，与 `ZONE_NAMES` 一一对应。
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive; got {width}x{height}.")

    x_edges = np.linspace(0, width, cols + 1)
    y_edges = np.linspace(0, height, rows + 1)

    polygons: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            polygons.append(
                np.array(
                    [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                    ],
                    dtype=np.int64,
                )
            )
    return polygons


def analyze_zones(
    detections: sv.Detections,
    width: int,
    height: int,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> ZoneAnalysis:
    """统计检测目标在九宫格各区域中的分布。

    Args:
        detections: Supervision 检测结果，使用检测框中心点作为锚点。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        rows: 网格行数，默认 3。
        cols: 网格列数，默认 3。

    Returns:
        `ZoneAnalysis`，包含每个区域的缺陷数量、类别细分、归属统计以及
        缺陷集中区域（hotspot）。
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive; got {width}x{height}.")

    zone_names = _zone_names(rows, cols)
    counts = {name: 0 for name in zone_names}
    class_counts: dict[str, dict[int, int]] = {name: {} for name in zone_names}

    total_detections = len(detections)
    if total_detections == 0:
        return ZoneAnalysis(counts=counts, class_counts=class_counts)

    polygons = build_grid_polygons(width, height, rows=rows, cols=cols)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
            require_all_anchors=True,
        )
        for polygon in polygons
    ]

    # 每个区域判断一次，得到 (区域数, 检测数) 的布尔矩阵
    hit_matrix = np.vstack([zone.trigger(detections) for zone in zones])

    # 中心点可能同时落在分割线两侧的多个区域里，按固定顺序取第一个命中项，
    # 保证每个检测框只归属一个区域，区域计数之和等于检测目标数。
    assigned = np.zeros(total_detections, dtype=bool)
    for detection_index in range(total_detections):
        for zone_index in range(len(zone_names)):
            if hit_matrix[zone_index, detection_index]:
                name = zone_names[zone_index]
                counts[name] += 1
                assigned[detection_index] = True

                class_id = _class_id_at(detections, detection_index)
                if class_id is not None:
                    bucket = class_counts[name]
                    bucket[class_id] = bucket.get(class_id, 0) + 1
                break

    return ZoneAnalysis(
        counts=counts,
        class_counts=class_counts,
        assigned_count=int(assigned.sum()),
        unassigned_count=int(total_detections - assigned.sum()),
    )


def _zone_names(rows: int, cols: int) -> tuple[str, ...]:
    if rows == GRID_ROWS and cols == GRID_COLS:
        return ZONE_NAMES
    return tuple(f"第{row + 1}行第{col + 1}列" for row in range(rows) for col in range(cols))


def _class_id_at(detections: sv.Detections, index: int) -> int | None:
    class_id = detections.class_id
    if class_id is None:
        return None
    value = int(class_id[index])
    return value
