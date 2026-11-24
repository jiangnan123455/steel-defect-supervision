"""中文标签渲染。

Supervision 的 `LabelAnnotator` 内部用 `cv2.putText` 绘制文字，
该函数只能渲染 ASCII，缺陷类别名（龟裂、夹杂等）会显示成 "??"。

本模块提供 `ChineseLabelAnnotator`：继承 `sv.LabelAnnotator`，沿用它的
定位、配色与背景框逻辑，只把文字绘制部分换成 PIL 的 TrueType 渲染，
从而在 Windows 上正确显示中文缺陷类别。

字体按 `CHINESE_FONT_CANDIDATES` 顺序查找；都找不到时降级为 ASCII 标签，
保证在无中文字体的系统上程序仍可运行。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Deng.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def find_chinese_font(size: int = 16) -> ImageFont.FreeTypeFont | None:
    """在系统候选路径中查找一个可用的中文字体。

    Args:
        size: 字号（像素）。

    Returns:
        字体对象；找不到时返回 None。
    """
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached

    for candidate in CHINESE_FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size)
        except OSError:
            continue
        _FONT_CACHE[size] = font
        return font

    return None


def clamp_label_boxes(
    boxes: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """把标签矩形整体平移进画面内，保持尺寸不变。

    框贴边时（例如缺陷框顶部压在图片上边缘），默认标签会画到画面外被裁掉。
    这里对标签矩形的左上角做钳制，保证完整可见。

    Args:
        boxes: 形状 `(N, >=4)` 的标签矩形，列为 `x1, y1, x2, y2`（可带额外列）。
        width: 画面宽度（像素）。
        height: 画面高度（像素）。

    Returns:
        调整后的副本，形状与原数组一致。
    """
    if boxes.size == 0:
        return boxes.copy()

    clamped = boxes.copy()
    for index in range(len(clamped)):
        x1, y1, x2, y2 = clamped[index, :4]
        box_width = x2 - x1
        box_height = y2 - y1

        # 先贴左上边界；若标签比画面还宽/高，贴住起点即可
        new_x1 = max(0.0, min(float(x1), float(width) - box_width))
        new_y1 = max(0.0, min(float(y1), float(height) - box_height))

        clamped[index, 0] = new_x1
        clamped[index, 1] = new_y1
        clamped[index, 2] = new_x1 + box_width
        clamped[index, 3] = new_y1 + box_height

    return clamped


class ChineseLabelAnnotator(sv.LabelAnnotator):
    """支持中文标签的 Supervision 标注器。

    与 `sv.LabelAnnotator` 的差别仅在于文字渲染方式：优先使用 PIL 的
    TrueType 字体绘制中文，字体不可用时降级为 ASCII。
    """

    def __init__(
        self,
        *args,
        font_path: str | None = None,
        fallback_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            font_path: 指定字体文件路径；为 None 时自动查找系统中文字体。
            fallback_only: 为 True 时只使用 ASCII 标签，不查找中文字体。
            其他参数与 `sv.LabelAnnotator` 一致。
        """
        super().__init__(*args, **kwargs)
        self._font_path = font_path
        self._fallback_only = fallback_only

    def annotate(
        self,
        scene,
        detections: sv.Detections,
        labels: list[str] | None = None,
        custom_color_lookup=None,
    ):
        """在图像上绘制带背景框的标签，文字使用中文字体渲染。"""
        if not isinstance(scene, np.ndarray):
            return scene

        if len(detections) == 0:
            return scene

        font = self._load_font()

        if font is None:
            # 没有可用中文字体时，退回父类的 ASCII 渲染路径。
            return super().annotate(scene, detections, labels, custom_color_lookup)

        if labels is None:
            labels = self._default_labels(detections)
        labels = list(labels)

        # 父类在绘制背景框时会原地修改 scene，这里先复制一份，
        # 保证调用方传入的原图不被改动。
        scene = scene.copy()

        label_properties = self._get_label_properties(detections, labels)
        # 父类按 cv2 字体的宽度估算背景框，中文用 PIL 渲染会更宽，
        # 这里按实际字形重新贴合，否则文字会超出底色被画面裁掉
        label_properties = self._fit_label_boxes(label_properties, labels, font)
        # 框贴边时把标签整体拉回画面内，避免文字被裁掉
        label_properties = clamp_label_boxes(
            label_properties, width=scene.shape[1], height=scene.shape[0]
        )
        color_lookup = (
            custom_color_lookup if custom_color_lookup is not None else self.color_lookup
        )

        # 第一趟：在 numpy 场景上画标签背景框（沿用父类的圆角矩形实现）。
        # 注意：numpy 数组是 BGR，父类 draw_rounded_rectangle 也接收 BGR，
        # 因此这里保持 as_bgr() 不变。
        text_colors: list[tuple[int, int, int]] = []
        for index, label_property in enumerate(label_properties):
            background_color = _resolve_color(
                self.color, detections, index, color_lookup
            )
            text_color = _resolve_color(
                self.text_color, detections, index, color_lookup
            )
            # PIL 只是把颜色数值原样写进数组，不做色彩空间转换，
            # 所以文字也用 BGR，避免与背景框出现通道不一致。
            text_colors.append(tuple(int(channel) for channel in text_color.as_bgr()))

            box_xyxy = label_property[:4].astype(int)
            self.draw_rounded_rectangle(
                scene=scene,
                xyxy=box_xyxy,
                color=background_color.as_bgr(),
                border_radius=self.border_radius,
            )

        # 第二趟：在背景框之上用 PIL 绘制中文文字。
        canvas = Image.fromarray(scene)
        drawer = ImageDraw.Draw(canvas)

        for index, label_property in enumerate(label_properties):
            box_xyxy = label_property[:4].astype(int)
            drawer.text(
                (box_xyxy[0] + self.text_padding, box_xyxy[1] + self.text_padding),
                labels[index],
                font=font,
                fill=text_colors[index],
            )

        # fromarray / array 之间没有真实色彩转换，直接写回即可；
        # 再叠加一次 ::-1 会把 numpy 画好的背景框通道反转。
        scene[:] = np.array(canvas)
        return scene

    def _fit_label_boxes(self, boxes, labels: list[str], font) -> np.ndarray:
        """按 PIL 实测的字形尺寸重设背景框，让底色正好包住文字。

        Args:
            boxes: 形状 ``(N, >=4)`` 的标签矩形，列为 ``x1, y1, x2, y2``。
            labels: 与矩形一一对应的标签文本。
            font: 已加载的中文字体。

        Returns:
            调整宽高后的副本，形状与原数组一致。
        """
        fitted = boxes.copy()
        for index, label in enumerate(labels):
            if index >= len(fitted):
                break
            left, top, right, bottom = font.getbbox(label)
            fitted[index, 2] = fitted[index, 0] + (right - left) + 2 * self.text_padding
            fitted[index, 3] = fitted[index, 1] + (bottom - top) + 2 * self.text_padding
        return fitted

    def _default_labels(self, detections: sv.Detections) -> list[str]:
        from supervision.detection.core import CLASS_NAME_DATA_FIELD

        class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        if class_names is None:
            return [str(index) for index in range(len(detections))]

        labels = []
        for index, name in enumerate(class_names):
            if detections.confidence is None:
                labels.append(str(name))
            else:
                labels.append(f"{name} {float(detections.confidence[index]):.2f}")
        return labels

    def _load_font(self) -> ImageFont.FreeTypeFont | None:
        if self._fallback_only:
            return None

        size = max(int(round(self.text_scale * 32)), 10)

        if self._font_path:
            path = Path(self._font_path)
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    return None
            return None

        return find_chinese_font(size)


def _resolve_color(color, detections: sv.Detections, index: int, color_lookup):
    """复用 Supervision 的颜色解析逻辑，保持与父类一致的配色。"""
    from supervision.annotators.core import _resolve_detection_color

    return _resolve_detection_color(
        color=color,
        detections=detections,
        detection_idx=index,
        color_lookup=color_lookup,
    )
