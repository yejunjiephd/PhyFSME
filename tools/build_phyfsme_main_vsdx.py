from __future__ import annotations

import math
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\code\Time_Series_Forecasting\paper6\PhyFSME")
SOURCE = ROOT / "主图.vsdx"
OUTPUT = ROOT / "主图-PhyFSME-最终版.vsdx"
PREVIEW = ROOT / "主图-PhyFSME-最终版-preview.png"

VNS = "http://schemas.microsoft.com/office/visio/2012/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NSMAP = {None: VNS, "r": RNS}

PAGE_W = 16.0
PAGE_H = 9.0


def ve(tag: str, **attrs) -> etree._Element:
    return etree.Element(f"{{{VNS}}}{tag}", **{k: str(v) for k, v in attrs.items() if v is not None})


def add_cell(parent: etree._Element, name: str, value, formula: str | None = None, unit: str | None = None) -> None:
    attrs = {"N": name, "V": str(value)}
    if formula:
        attrs["F"] = formula
    if unit:
        attrs["U"] = unit
    parent.append(ve("Cell", **attrs))


class VBuilder:
    def __init__(self) -> None:
        self.next_id = 1000
        self.shapes = ve("Shapes")

    def _id(self) -> str:
        self.next_id += 1
        return str(self.next_id)

    def add_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        fill: str,
        line: str = "#2E3A46",
        font: str = "#1F2933",
        size_pt: float = 10.5,
        bold: bool = False,
        master: str = "2",
        weight: float = 1.15,
    ) -> str:
        s = ve("Shape", ID=self._id(), NameU="Rounded Rectangle", Name="Rounded Rectangle", Type="Shape", Master=master)
        for n, v in [
            ("PinX", x),
            ("PinY", y),
            ("Width", w),
            ("Height", h),
            ("LocPinX", w / 2),
            ("LocPinY", h / 2),
            ("FillForegnd", fill),
            ("FillBkgnd", fill),
            ("FillPattern", 1),
            ("LineColor", line),
            ("LineWeight", weight / 72),
            ("ShapeShdwShow", 2),
            ("FillGradientEnabled", 0),
            ("TxtPinX", w / 2),
            ("TxtPinY", h / 2),
            ("TxtWidth", w * 0.92),
            ("TxtHeight", h * 0.78),
        ]:
            add_cell(s, n, v)
        char = ve("Section", N="Character")
        row = ve("Row", IX="0")
        add_cell(row, "Font", "Aptos")
        add_cell(row, "Size", size_pt / 72, unit="PT")
        add_cell(row, "Color", font)
        if bold:
            add_cell(row, "Style", 17)
        char.append(row)
        s.append(char)
        para = ve("Section", N="Paragraph")
        prow = ve("Row", IX="0")
        add_cell(prow, "HorzAlign", 1)
        para.append(prow)
        s.append(para)
        tb = ve("Section", N="TextBlock")
        tbr = ve("Row", IX="0")
        add_cell(tbr, "VerticalAlign", 1)
        tb.append(tbr)
        s.append(tb)
        text_el = ve("Text")
        cp = ve("cp", IX="0")
        text_el.append(cp)
        cp.tail = text
        s.append(text_el)
        self.shapes.append(s)
        return s.get("ID")

    def add_rect_panel(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        line: str,
        title: str,
        title_color: str,
    ) -> None:
        self.add_box(x, y, w, h, "", fill, line, size_pt=8, weight=1.0)
        self.add_text(x - w / 2 + 0.34, y + h / 2 - 0.32, title, 11.0, title_color, bold=True, align=0)

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        size_pt: float = 10,
        color: str = "#1F2933",
        bold: bool = False,
        w: float = 2.0,
        h: float = 0.35,
        align: int = 1,
    ) -> str:
        s = ve("Shape", ID=self._id(), Type="Shape")
        for n, v in [
            ("PinX", x),
            ("PinY", y),
            ("Width", w),
            ("Height", h),
            ("LocPinX", w / 2),
            ("LocPinY", h / 2),
            ("FillPattern", 0),
            ("LinePattern", 0),
            ("LineWeight", 0),
            ("TxtPinX", w / 2),
            ("TxtPinY", h / 2),
            ("TxtWidth", w),
            ("TxtHeight", h),
        ]:
            add_cell(s, n, v)
        geom = ve("Section", N="Geometry", IX="0")
        add_cell(geom, "NoFill", 1)
        add_cell(geom, "NoLine", 1)
        s.append(geom)
        char = ve("Section", N="Character")
        row = ve("Row", IX="0")
        add_cell(row, "Font", "Aptos")
        add_cell(row, "Size", size_pt / 72, unit="PT")
        add_cell(row, "Color", color)
        if bold:
            add_cell(row, "Style", 17)
        char.append(row)
        s.append(char)
        para = ve("Section", N="Paragraph")
        prow = ve("Row", IX="0")
        add_cell(prow, "HorzAlign", align)
        para.append(prow)
        s.append(para)
        text_el = ve("Text")
        cp = ve("cp", IX="0")
        text_el.append(cp)
        cp.tail = text
        s.append(text_el)
        self.shapes.append(s)
        return s.get("ID")

    def add_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#3B4A5A",
        weight_pt: float = 1.7,
        end: bool = True,
        dash: int | None = None,
    ) -> str:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        s = ve("Shape", ID=self._id(), Type="Shape", LineStyle="3", FillStyle="3", TextStyle="3")
        cells = [
            ("PinX", (x1 + x2) / 2, "(BeginX+EndX)/2"),
            ("PinY", (y1 + y2) / 2, "(BeginY+EndY)/2"),
            ("Width", length, "SQRT((EndX-BeginX)^2+(EndY-BeginY)^2)"),
            ("Height", 0, None),
            ("LocPinX", length / 2, "Width*0.5"),
            ("LocPinY", 0, "Height*0.5"),
            ("Angle", math.atan2(dy, dx), "ATAN2(EndY-BeginY,EndX-BeginX)"),
            ("BeginX", x1, None),
            ("BeginY", y1, None),
            ("EndX", x2, None),
            ("EndY", y2, None),
            ("LineColor", color, None),
            ("LineWeight", weight_pt / 72, None),
        ]
        for n, v, f in cells:
            add_cell(s, n, v, formula=f)
        if end:
            add_cell(s, "EndArrow", 5)
        if dash is not None:
            add_cell(s, "LinePattern", dash)
        geom = ve("Section", N="Geometry", IX="0")
        add_cell(geom, "NoFill", 1)
        add_cell(geom, "NoLine", 0)
        r1 = ve("Row", T="MoveTo", IX="1")
        add_cell(r1, "X", 0, "Width*0")
        add_cell(r1, "Y", 0)
        r2 = ve("Row", T="LineTo", IX="2")
        add_cell(r2, "X", length, "Width*1")
        add_cell(r2, "Y", 0)
        geom.extend([r1, r2])
        s.append(geom)
        self.shapes.append(s)
        return s.get("ID")

    def add_polyline(self, points: list[tuple[float, float]], color: str, weight_pt: float = 1.2) -> None:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        w, h = max(maxx - minx, 0.01), max(maxy - miny, 0.01)
        s = ve("Shape", ID=self._id(), Type="Shape")
        for n, v in [
            ("PinX", (minx + maxx) / 2),
            ("PinY", (miny + maxy) / 2),
            ("Width", w),
            ("Height", h),
            ("LocPinX", w / 2),
            ("LocPinY", h / 2),
            ("LineColor", color),
            ("LineWeight", weight_pt / 72),
        ]:
            add_cell(s, n, v)
        geom = ve("Section", N="Geometry", IX="0")
        add_cell(geom, "NoFill", 1)
        add_cell(geom, "NoLine", 0)
        for ix, (px, py) in enumerate(points, start=1):
            row = ve("Row", T="MoveTo" if ix == 1 else "LineTo", IX=str(ix))
            add_cell(row, "X", px - minx)
            add_cell(row, "Y", py - miny)
            geom.append(row)
        s.append(geom)
        self.shapes.append(s)

    def page_xml(self) -> bytes:
        page = etree.Element(f"{{{VNS}}}PageContents", nsmap=NSMAP)
        page.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        page.append(self.shapes)
        page.append(ve("Connects"))
        return etree.tostring(page, xml_declaration=True, encoding="utf-8", pretty_print=False)


def build_shapes() -> VBuilder:
    b = VBuilder()

    # Background and title band
    b.add_box(8, 4.5, 15.7, 8.55, "", "#F8FAFC", "#D9E2EC", size_pt=1, weight=0.7)
    b.add_text(
        8,
        8.55,
        "PhyFSME: Physics-Consistent Multi-Scale Fractional Spectral Mixture-of-Experts",
        18,
        "#0B1F33",
        True,
        w=12.6,
        h=0.45,
    )
    b.add_text(
        8,
        8.16,
        "可学习时频旋转  |  多尺度谱专家路由  |  复数域稳定性正则化  |  跨变量交互融合",
        9.6,
        "#52616F",
        False,
        w=10.6,
        h=0.32,
    )

    # Left input block
    b.add_box(1.15, 4.82, 1.65, 1.05, "Input\nX ∈ R^{L×C}", "#E8F4FF", "#2F80ED", "#12355B", 10.8, True)
    b.add_text(1.15, 4.05, "multivariate\nlookback window", 7.8, "#52616F", w=1.5, h=0.5)
    # Mini signal traces
    for j, c in enumerate(["#2F80ED", "#00A6A6", "#F59E0B"]):
        pts = []
        for k in range(26):
            x = 0.55 + k * 0.046
            y = 5.47 - j * 0.13 + math.sin(k * 0.65 + j) * 0.035
            pts.append((x, y))
        b.add_polyline(pts, c, 1.1)
    b.add_arrow(2.05, 4.82, 2.65, 4.82, "#3B82F6", 1.8)
    b.add_box(3.28, 4.82, 1.24, 0.72, "Instance\nNorm", "#EEF7ED", "#3A9D5D", "#164B2F", 9.5, True)

    # Spectral stream panel
    b.add_rect_panel(7.25, 5.78, 7.55, 3.42, "#FFFFFF", "#C9D6E2", "A  Fractional Spectral Stream", "#0B5E7A")
    b.add_text(7.25, 7.2, "Multi-scale learnable FRFT experts", 12.2, "#0B5E7A", True, w=3.7, h=0.35)
    y_rows = [6.62, 5.78, 4.94]
    scale_labels = ["Scale-1  P=L", "Scale-2  P=L/4", "Scale-3  P=L/8"]
    alpha_labels = ["α₁", "α₂", "α₃"]
    b.add_arrow(3.9, 4.98, 4.0, 6.62, "#3A9D5D", 1.25, dash=2)
    for idx, y in enumerate(y_rows):
        b.add_box(4.45, y, 1.25, 0.46, scale_labels[idx], "#ECFDF5", "#10B981", "#065F46", 8.2, True)
        b.add_arrow(5.08, y, 5.45, y, "#0F766E", 1.25)
        b.add_box(6.05, y, 1.14, 0.54, f"FRFT\n{alpha_labels[idx]}", "#E0F2FE", "#0284C7", "#075985", 9.0, True)
        # small rotated basis symbol
        b.add_polyline([(5.73, y - 0.17), (5.86, y + 0.14), (6.02, y - 0.06), (6.2, y + 0.16), (6.37, y - 0.12)], "#0284C7", 1.0)
        b.add_arrow(6.63, y, 6.98, y, "#0F766E", 1.25)
        b.add_box(7.62, y, 1.18, 0.54, "Adaptive\nMask", "#FFF7ED", "#F97316", "#9A3412", 8.6, True)
        b.add_arrow(8.22, y, 8.57, y, "#0F766E", 1.25)
        b.add_box(9.25, y, 1.18, 0.54, "Complex\nMixer", "#FEF2F2", "#EF4444", "#7F1D1D", 8.6, True)
        b.add_arrow(9.86, y, 10.22, y, "#0F766E", 1.25)
        b.add_box(10.85, y, 1.08, 0.54, "iFRFT\n−α", "#E0F2FE", "#0284C7", "#075985", 8.6, True)
    b.add_box(11.28, 5.78, 0.92, 2.25, "Scale\nMoE\nRouter", "#F0FDFA", "#14B8A6", "#115E59", 9.2, True)
    for y in y_rows:
        b.add_arrow(11.39, y, 10.92, y, "#0F766E", 1.2, end=False)
    b.add_arrow(11.75, 5.78, 12.28, 5.26, "#14B8A6", 1.7)
    b.add_text(12.02, 5.95, "z_spec", 7.8, "#0891B2", w=0.7, h=0.22)
    b.add_text(7.45, 4.22, "Energy focusing in a learnable time-frequency coordinate", 8.4, "#64748B", w=5.8, h=0.25)

    # Variable interaction stream panel
    b.add_rect_panel(7.25, 2.55, 7.55, 2.15, "#FFFFFF", "#C9D6E2", "B  Variable Interaction Stream", "#6B4E16")
    b.add_arrow(3.9, 4.66, 4.0, 2.55, "#3A9D5D", 1.25, dash=2)
    b.add_box(4.7, 2.55, 1.45, 0.62, "Variable\nTokens", "#FEFCE8", "#CA8A04", "#713F12", 9.0, True)
    b.add_arrow(5.43, 2.55, 6.05, 2.55, "#CA8A04", 1.6)
    b.add_box(6.82, 2.55, 1.45, 0.62, "Multi-head\nAttention", "#FFFBEB", "#D97706", "#78350F", 9.0, True)
    b.add_arrow(7.55, 2.55, 8.15, 2.55, "#CA8A04", 1.6)
    b.add_box(8.82, 2.55, 1.12, 0.62, "FFN +\nResidual", "#FFF7ED", "#EA580C", "#7C2D12", 8.8, True)
    b.add_arrow(9.42, 2.55, 10.1, 2.55, "#CA8A04", 1.6)
    b.add_box(10.88, 2.55, 1.42, 0.62, "Cross-variable\nFeatures", "#FFF7ED", "#EA580C", "#7C2D12", 8.4, True)
    b.add_arrow(11.62, 2.55, 12.28, 3.85, "#D97706", 1.5, dash=2)
    b.add_text(12.0, 3.34, "z_var", 7.8, "#D97706", w=0.7, h=0.22)
    b.add_text(7.25, 1.7, "Global dependency modeling across channels C", 8.4, "#64748B", w=4.2, h=0.25)

    # Fusion and prediction
    b.add_rect_panel(13.75, 4.42, 3.35, 2.75, "#FFFFFF", "#B6C2CF", "C  Dynamic Fusion & Prediction", "#334155")
    b.add_box(12.7, 4.55, 1.18, 0.78, "Gated\nFusion", "#EEF2FF", "#4F46E5", "#312E81", 9.0, True)
    b.add_arrow(13.3, 4.55, 13.72, 4.55, "#4F46E5", 1.7)
    b.add_box(14.08, 4.55, 0.7, 0.78, "Linear\nHead", "#F1F5F9", "#475569", "#1E293B", 7.7, True)
    b.add_arrow(14.45, 4.55, 14.82, 4.55, "#475569", 1.6)
    b.add_box(15.25, 4.55, 0.86, 0.78, "Output\nŶ ∈ R^{H×C}", "#E8F4FF", "#2F80ED", "#12355B", 7.7, True)
    b.add_text(13.92, 3.37, "z = g⊙z_spec + (1−g)⊙z_var", 8.0, "#4F46E5", w=2.5, h=0.3)

    # Regularization block
    b.add_rect_panel(7.55, 0.88, 7.55, 1.35, "#F8FAFC", "#D9E2EC", "D  Complex-domain Stability Regularizer", "#991B1B")
    b.add_box(4.15, 0.88, 1.55, 0.48, "Imag-energy\nratio", "#FEF2F2", "#EF4444", "#7F1D1D", 8.2, True)
    b.add_box(6.16, 0.88, 1.55, 0.48, "Temporal\nsmoothness", "#FFF7ED", "#F97316", "#9A3412", 8.2, True)
    b.add_box(8.17, 0.88, 1.55, 0.48, "Magnitude\nsparsity", "#FFFBEB", "#EAB308", "#713F12", 8.2, True)
    b.add_text(10.45, 0.88, "ℒ = ℒ_pred + Σᵢ(β₁ℒ_ratio + β₂ℒ_smooth + β₃ℒ_mag)", 9.0, "#991B1B", True, w=4.0, h=0.38)
    b.add_arrow(8.65, 4.58, 8.65, 1.55, "#EF4444", 1.1, end=True, dash=2)

    # Small callouts
    b.add_box(2.15, 7.2, 2.55, 0.62, "Problem: fixed Fourier bases blur\nchirp-like non-stationary patterns", "#F8FAFC", "#94A3B8", "#334155", 8.3, False)
    b.add_arrow(3.43, 6.94, 5.35, 6.68, "#94A3B8", 1.1, dash=2)
    b.add_box(12.3, 7.23, 2.65, 0.62, "Key idea: learn α and route\nscales according to input content", "#F8FAFC", "#94A3B8", "#334155", 8.3, False)
    b.add_arrow(12.0, 6.95, 11.85, 6.2, "#94A3B8", 1.1, dash=2)

    return b


def update_pages_xml(xml_bytes: bytes) -> bytes:
    root = etree.fromstring(xml_bytes)
    ns = {"v": VNS}
    page = root.find(f"{{{VNS}}}Page")
    page.set("Name", "PhyFSME-main-final")
    page.set("NameU", "PhyFSME-main-final")
    page.set("ViewScale", "1")
    page.set("ViewCenterX", str(PAGE_W / 2))
    page.set("ViewCenterY", str(PAGE_H / 2))
    sheet = page.find(f"{{{VNS}}}PageSheet")
    for cell in sheet.findall(f"{{{VNS}}}Cell"):
        if cell.get("N") == "PageWidth":
            cell.set("V", str(PAGE_W))
        elif cell.get("N") == "PageHeight":
            cell.set("V", str(PAGE_H))
        elif cell.get("N") in {"XRulerOrigin", "XGridOrigin"}:
            cell.set("V", "0")
        elif cell.get("N") in {"YRulerOrigin", "YGridOrigin"}:
            cell.set("V", str(PAGE_H))
    return etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=False)


def build_vsdx() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    builder = build_shapes()
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(SOURCE, "r") as zin, zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "visio/pages/page1.xml":
                data = builder.page_xml()
            elif item.filename == "visio/pages/pages.xml":
                data = update_pages_xml(data)
            zout.writestr(item, data)


# Lightweight preview for iteration; final deliverable remains VSDX.
def draw_preview() -> None:
    W, H = 2400, 1350
    sx, sy = W / PAGE_W, H / PAGE_H
    img = Image.new("RGB", (W, H), "#F8FAFC")
    d = ImageDraw.Draw(img)

    def xy(x, y):
        return x * sx, H - y * sy

    def rect(x, y, w, h, fill, outline, r=18, width=3):
        x1, y1 = xy(x - w / 2, y + h / 2)
        x2, y2 = xy(x + w / 2, y - h / 2)
        d.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill, outline=outline, width=width)

    def arrow(x1, y1, x2, y2, fill="#334155", width=4):
        d.line([xy(x1, y1), xy(x2, y2)], fill=fill, width=width)
        ang = math.atan2((H - y2 * sy) - (H - y1 * sy), x2 * sx - x1 * sx)
        end = xy(x2, y2)
        size = 14
        pts = [
            end,
            (end[0] - size * math.cos(ang - 0.45), end[1] - size * math.sin(ang - 0.45)),
            (end[0] - size * math.cos(ang + 0.45), end[1] - size * math.sin(ang + 0.45)),
        ]
        d.polygon(pts, fill=fill)

    try:
        font_big = ImageFont.truetype("arial.ttf", 42)
        font_mid = ImageFont.truetype("arial.ttf", 26)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_big = font_mid = font_small = None

    d.rounded_rectangle([18, 18, W - 18, H - 18], radius=26, fill="#F8FAFC", outline="#D9E2EC", width=3)
    d.text((W / 2, 52), "PhyFSME: Physics-Consistent Multi-Scale Fractional Spectral Mixture-of-Experts", fill="#0B1F33", font=font_big, anchor="ma")
    d.text((W / 2, 115), "learnable time-frequency rotation | scale-aware routing | complex-domain stability | variable interaction", fill="#52616F", font=font_small, anchor="ma")

    rect(1.15, 4.82, 1.65, 1.05, "#E8F4FF", "#2F80ED")
    d.text(xy(1.15, 4.95), "Input\nX", fill="#12355B", font=font_mid, anchor="mm", align="center")
    rect(3.28, 4.82, 1.24, 0.72, "#EEF7ED", "#3A9D5D")
    d.text(xy(3.28, 4.9), "Instance\nNorm", fill="#164B2F", font=font_small, anchor="mm", align="center")
    arrow(2.05, 4.82, 2.65, 4.82, "#3B82F6")

    rect(7.35, 5.78, 7.15, 3.42, "#FFFFFF", "#C9D6E2", r=24)
    d.text(xy(4.1, 7.33), "A Fractional Spectral Stream", fill="#0B5E7A", font=font_small, anchor="lm")
    for y in [6.62, 5.78, 4.94]:
        for x, label, col in [(4.45, "Scale", "#ECFDF5"), (6.05, "FRFT α", "#E0F2FE"), (7.62, "Mask", "#FFF7ED"), (9.25, "Mixer", "#FEF2F2"), (10.85, "iFRFT", "#E0F2FE")]:
            rect(x, y, 1.15, 0.54, col, "#64748B", r=14, width=2)
            d.text(xy(x, y + 0.03), label, fill="#1E293B", font=font_small, anchor="mm")
        for x1, x2 in [(5.08, 5.45), (6.63, 6.98), (8.22, 8.57), (9.86, 10.22)]:
            arrow(x1, y, x2, y, "#0F766E", width=3)
    rect(11.95, 5.78, 0.95, 2.25, "#F0FDFA", "#14B8A6")
    d.text(xy(11.95, 5.9), "Scale\nMoE", fill="#115E59", font=font_small, anchor="mm", align="center")
    arrow(12.42, 5.78, 13.1, 5.78, "#14B8A6")
    rect(13.75, 5.78, 1.15, 0.76, "#ECFEFF", "#0891B2")
    d.text(xy(13.75, 5.86), "Spectral\nFeatures", fill="#155E75", font=font_small, anchor="mm", align="center")

    rect(7.35, 2.55, 7.15, 2.15, "#FFFFFF", "#C9D6E2", r=24)
    d.text(xy(4.1, 3.43), "B Variable Interaction Stream", fill="#6B4E16", font=font_small, anchor="lm")
    for x, label in [(4.7, "Variable\nTokens"), (6.82, "MHA"), (8.82, "FFN"), (10.88, "Cross-var\nFeatures")]:
        rect(x, 2.55, 1.35, 0.62, "#FFFBEB", "#D97706")
        d.text(xy(x, 2.63), label, fill="#713F12", font=font_small, anchor="mm", align="center")
    for x1, x2 in [(5.43, 6.05), (7.55, 8.15), (9.42, 10.1)]:
        arrow(x1, 2.55, x2, 2.55, "#CA8A04")

    rect(13.55, 4.42, 2.35, 2.75, "#FFFFFF", "#B6C2CF", r=24)
    d.text(xy(12.5, 5.55), "C Dynamic Fusion", fill="#334155", font=font_small, anchor="lm")
    rect(13.55, 4.55, 1.45, 0.78, "#EEF2FF", "#4F46E5")
    d.text(xy(13.55, 4.63), "Gated\nFusion", fill="#312E81", font=font_small, anchor="mm", align="center")
    rect(15.16, 4.55, 0.76, 0.78, "#F1F5F9", "#475569")
    d.text(xy(15.16, 4.63), "Head", fill="#1E293B", font=font_small, anchor="mm")
    rect(15.85, 4.55, 0.88, 0.78, "#E8F4FF", "#2F80ED")
    d.text(xy(15.85, 4.63), "Output", fill="#12355B", font=font_small, anchor="mm")
    arrow(14.27, 4.55, 14.78, 4.55, "#4F46E5")
    arrow(15.55, 4.55, 15.85, 4.55, "#475569")

    rect(7.55, 0.88, 7.55, 1.35, "#F8FAFC", "#D9E2EC", r=24)
    d.text(xy(4.0, 1.42), "D Complex-domain Stability Regularizer", fill="#991B1B", font=font_small, anchor="lm")
    for x, label, fill, out in [(4.15, "Imag\nratio", "#FEF2F2", "#EF4444"), (6.16, "Smooth", "#FFF7ED", "#F97316"), (8.17, "Sparse", "#FFFBEB", "#EAB308")]:
        rect(x, 0.88, 1.55, 0.48, fill, out)
        d.text(xy(x, 0.94), label, fill="#7F1D1D", font=font_small, anchor="mm", align="center")
    d.text(xy(10.45, 0.92), "L = L_pred + Σ regularizers", fill="#991B1B", font=font_small, anchor="mm")

    img.save(PREVIEW)


if __name__ == "__main__":
    build_vsdx()
    draw_preview()
    print(f"Wrote {OUTPUT}")
    print(f"Preview {PREVIEW}")
