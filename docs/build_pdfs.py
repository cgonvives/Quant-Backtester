"""Convierte las fuentes Markdown de docs/fuentes/ en los PDF de docs/.

    python docs/build_pdfs.py          # todos
    python docs/build_pdfs.py 02 A     # solo los que empiezan por "02" o por "A"

Necesita requirements-dev.txt (reportlab). Las fórmulas se renderizan con matplotlib mathtext
y se guardan en docs/_build/ (caché, no se versiona).

Dialecto Markdown admitido (un subconjunto, a propósito):
- `# Título` (uno por fichero), `## Sección`, `### Subsección`
- párrafos; en línea: **negrita**, *cursiva*, `código`
- listas de un nivel: `- elemento` o `1. elemento`
- bloques de código con ```lenguaje
- fórmulas en bloque, en una línea:  $$ r_t = \\frac{P_t}{P_{t-1}} - 1 $$
- tablas pipe:  | a | b |  +  |---|---|
- avisos:  > [!PISTA] Título  /  > [!AVISO]  /  > [!ERROR]  /  > [!NOTA]   (líneas siguientes con "> ")
- figuras:  ![Pie de figura](fig:nombre)   → docs/fuentes/figuras/nombre.png
- salto de página: una línea con  \\pagebreak
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.mathtext import math_to_image  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable  # noqa: E402

DOCS = Path(__file__).resolve().parent
SOURCES = DOCS / "fuentes"
FIGURES = SOURCES / "figuras"
BUILD = DOCS / "_build"

PAGE_W, PAGE_H = A4
MARGIN = 2.1 * cm
TEXT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#e1e0d9")
CODE_BG = colors.HexColor("#f4f3ef")
CALLOUTS = {  # tipo: (etiqueta por defecto, color del borde, fondo)
    "PISTA": ("Pista", colors.HexColor("#2a78d6"), colors.HexColor("#eef4fc")),
    "AVISO": ("Aviso", colors.HexColor("#c98500"), colors.HexColor("#fdf5e1")),
    "ERROR": ("Error típico", colors.HexColor("#d03b3b"), colors.HexColor("#fcecec")),
    "NOTA": ("Nota", colors.HexColor("#898781"), colors.HexColor("#f4f3ef")),
}


# ---------------------------------------------------------------------------
# Fuentes y estilos
# ---------------------------------------------------------------------------
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = {
    "DejaVu": "DejaVuSans.ttf",
    "DejaVu-Bold": "DejaVuSans-Bold.ttf",
    "DejaVu-Oblique": "DejaVuSans-Oblique.ttf",
    "DejaVu-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    "DejaVuMono": "DejaVuSansMono.ttf",
    "DejaVuMono-Bold": "DejaVuSansMono-Bold.ttf",
}


def register_fonts() -> None:
    for name, file in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique"
    )
    pdfmetrics.registerFontFamily(
        "DejaVuMono", normal="DejaVuMono", bold="DejaVuMono-Bold", italic="DejaVuMono", boldItalic="DejaVuMono-Bold"
    )


def make_styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle("body", fontName="DejaVu", fontSize=10, leading=14.6, textColor=INK, spaceAfter=6,
                          alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", parent=body, fontName="DejaVu-Bold", fontSize=22, leading=27, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=9.5, textColor=INK_2, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=15, leading=19, spaceBefore=16,
                             spaceAfter=6, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=body, fontName="DejaVu-Bold", fontSize=12, leading=15.5, spaceBefore=12,
                             spaceAfter=4, textColor=ACCENT, keepWithNext=1),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=3),
        "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=8.4, leading=11.4, textColor=INK),
        "caption": ParagraphStyle("caption", parent=body, fontName="DejaVu-Oblique", fontSize=8.6, leading=11.5,
                                  textColor=INK_2, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11.2, spaceAfter=0),
        "cell_head": ParagraphStyle("cell_head", parent=body, fontName="DejaVu-Bold", fontSize=8.6, leading=11.2,
                                    spaceAfter=0),
        "callout_title": ParagraphStyle("callout_title", parent=body, fontName="DejaVu-Bold", fontSize=9.4,
                                        leading=12.5, spaceAfter=2),
        "callout": ParagraphStyle("callout", parent=body, fontSize=9.4, leading=13.2, spaceAfter=3),
        "callout_bullet": ParagraphStyle("callout_bullet", parent=body, fontSize=9.4, leading=13.2, spaceAfter=2,
                                         leftIndent=14, bulletIndent=2),
    }


# ---------------------------------------------------------------------------
# Markdown → bloques
# ---------------------------------------------------------------------------
@dataclass
class Block:
    kind: str
    text: str = ""
    lang: str = ""
    items: list | None = None
    ordered: bool = False
    title: str = ""


LIST_ITEM = re.compile(r"^\s*(?:[-*]|(\d+)\.)\s+(.*)$")
IMAGE = re.compile(r"^!\[(.*)\]\((.+)\)\s*$")
CALLOUT_HEAD = re.compile(r"^\[!(\w+)\]\s*(.*)$")


def starts_block(line: str) -> bool:
    s = line.strip()
    return (
        s.startswith(("#", "```", "$$", ">", "|", "![")) or s == r"\pagebreak" or bool(LIST_ITEM.match(line))
    )


def parse(md: str, source: str) -> list[Block]:
    lines = md.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
        elif s.startswith("```"):
            lang, body = s[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError(f"{source}: bloque de código sin cerrar")
            blocks.append(Block("code", "\n".join(body), lang=lang))
            i += 1
        elif s == r"\pagebreak":
            blocks.append(Block("pagebreak"))
            i += 1
        elif s.startswith("### "):
            blocks.append(Block("h3", s[4:]))
            i += 1
        elif s.startswith("## "):
            blocks.append(Block("h2", s[3:]))
            i += 1
        elif s.startswith("# "):
            blocks.append(Block("h1", s[2:]))
            i += 1
        elif s.startswith("$$"):
            parts = [s]
            while not (parts[-1].endswith("$$") and len("".join(parts)) > 4):
                i += 1
                if i == len(lines):
                    raise ValueError(f"{source}: fórmula $$ sin cerrar")
                parts.append(lines[i].strip())
            blocks.append(Block("math", " ".join(parts)[2:-2].strip()))
            i += 1
        elif s.startswith(">"):
            body = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                body.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            head = CALLOUT_HEAD.match(body[0].strip()) if body else None
            kind, title = ("NOTA", "")
            if head:
                kind, title = head.group(1).upper(), head.group(2).strip()
                body = body[1:]
            if kind not in CALLOUTS:
                raise ValueError(f"{source}: tipo de aviso desconocido [!{kind}]")
            blocks.append(Block("callout", "\n".join(body), lang=kind, title=title))
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(Block("table", items=rows))
        elif IMAGE.match(s):
            m = IMAGE.match(s)
            blocks.append(Block("image", m.group(1), lang=m.group(2)))
            i += 1
        elif LIST_ITEM.match(line):
            items, ordered = [], bool(LIST_ITEM.match(line).group(1))
            while i < len(lines) and lines[i].strip():
                m = LIST_ITEM.match(lines[i])
                if m:
                    items.append(m.group(2).strip())
                elif starts_block(lines[i]):
                    break
                else:
                    items[-1] += " " + lines[i].strip()  # continuación del elemento anterior
                i += 1
            blocks.append(Block("list", items=items, ordered=ordered))
        else:
            para = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
                para.append(lines[i].strip())
                i += 1
            blocks.append(Block("para", " ".join(para)))
    return blocks


def inline(text: str) -> str:
    """Markdown en línea → mini-HTML de reportlab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes: list[str] = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return re.sub(
        r"\x00(\d+)\x00",
        lambda m: f'<font face="DejaVuMono" size="8.8" color="#7a3410" backColor="#f4f3ef">'
        f"{codes[int(m.group(1))]}</font>",
        text,
    )


# ---------------------------------------------------------------------------
# Bloques → flowables
# ---------------------------------------------------------------------------
def formula(tex: str, source: str):
    key = hashlib.sha1(tex.encode("utf-8")).hexdigest()[:16]
    path = BUILD / "formulas" / f"{key}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            math_to_image(f"${tex}$", str(path), prop=FontProperties(size=13), dpi=300, format="png", color="#0b0b0b")
        except Exception as exc:
            raise ValueError(f"{source}: no se puede renderizar la fórmula\n  {tex}\n  {exc}") from exc
    with PILImage.open(path) as im:
        w, h = im.size
    width, height = w / 300 * 72, h / 300 * 72
    if width > TEXT_W:
        width, height = TEXT_W, height * TEXT_W / width
    img = Image(str(path), width=width, height=height)
    img.hAlign = "CENTER"
    return [Spacer(1, 3), img, Spacer(1, 7)]


def figure(caption: str, target: str, styles, source: str):
    if not target.startswith("fig:"):
        raise ValueError(f"{source}: solo se admiten figuras fig:nombre (recibido {target})")
    path = FIGURES / f"{target[4:]}.png"
    if not path.exists():
        raise FileNotFoundError(f"{source}: no existe la figura {path}")
    with PILImage.open(path) as im:
        w, h = im.size
    width = min(TEXT_W, 16 * cm)
    img = Image(str(path), width=width, height=width * h / w)
    img.hAlign = "CENTER"
    return [KeepTogether([Spacer(1, 4), img, Paragraph(inline(caption), styles["caption"])])]


def code_block(text: str, styles):
    style = styles["code"]
    chars_per_line = int((TEXT_W - 16) / pdfmetrics.stringWidth("M", style.fontName, style.fontSize))
    pre = Preformatted(text, style, maxLineLength=chars_per_line, newLineChars="  ")
    box = Table([[pre]], colWidths=[TEXT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [box, Spacer(1, 8)]


def callout(block: Block, styles):
    label, edge, background = CALLOUTS[block.lang]
    content = [Paragraph(f'<font color="{edge.hexval().replace("0x", "#")}">{inline(block.title or label)}</font>',
                         styles["callout_title"])]
    for chunk in re.split(r"\n\s*\n", block.text.strip()):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        paragraph: list[str] = []
        for ln in lines:
            m = LIST_ITEM.match(ln)
            if m:
                if paragraph:
                    content.append(Paragraph(inline(" ".join(paragraph)), styles["callout"]))
                    paragraph = []
                bullet = f"{m.group(1)}." if m.group(1) else "•"
                content.append(Paragraph(inline(m.group(2)), styles["callout_bullet"], bulletText=bullet))
            else:
                paragraph.append(ln)
        if paragraph:
            content.append(Paragraph(inline(" ".join(paragraph)), styles["callout"]))
    box = Table([[content]], colWidths=[TEXT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [box, Spacer(1, 8)]


def text_width(md_text: str, bold: bool = False) -> float:
    """Ancho de un texto Markdown en línea tal y como se dibuja en una celda."""
    width = 0.0
    for k, part in enumerate(md_text.split("`")):
        font, size = ("DejaVuMono", 8.8) if k % 2 else ("DejaVu-Bold" if bold else "DejaVu", 8.6)
        width += pdfmetrics.stringWidth(part.replace("**", "").replace("*", ""), font, size)
    return width


def column_widths(rows: list[list[str]], padding: float = 14) -> list[float]:
    """Ancho natural de cada columna; si no cabe, se reparte sin partir ninguna palabra."""
    n = len(rows[0])

    def longest_word(text: str, bold: bool) -> float:
        return max(text_width(word, bold) for word in (text.split() or [""]))

    def widths_of(j, measure):
        return [measure(r[j], bold=(i == 0)) for i, r in enumerate(rows)]  # fila 0 = cabecera en negrita

    natural = [max(widths_of(j, text_width)) + padding for j in range(n)]
    minimum = [max(widths_of(j, longest_word)) + padding for j in range(n)]
    if sum(natural) <= TEXT_W:
        return [w * TEXT_W / sum(natural) for w in natural]
    spare = max(TEXT_W - sum(minimum), 0)
    extra = [nat - mn for nat, mn in zip(natural, minimum)]
    return [mn + spare * e / (sum(extra) or 1) for mn, e in zip(minimum, extra)]


def table(rows: list[str], styles, source: str):
    def cells(row):
        return [c.strip() for c in row.strip().strip("|").split("|")]

    if len(rows) < 2 or not re.match(r"^\|?\s*:?-{2,}", rows[1]):
        raise ValueError(f"{source}: tabla sin fila separadora |---|: {rows[0]}")
    header, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in body]
    widths = column_widths([header, *body])
    data = [[Paragraph(inline(c), styles["cell_head"]) for c in header]]
    data += [[Paragraph(inline(c), styles["cell"]) for c in r] for r in body]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CODE_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 9)]


def to_flowables(blocks: list[Block], styles, source: str) -> tuple[str, list]:
    title, story = "", []
    for b in blocks:
        if b.kind == "h1":
            title = b.text
            story += [
                Paragraph(inline(b.text), styles["title"]),
                Paragraph("Quant-Backtester · material de estudio", styles["subtitle"]),
                HRFlowable(width="100%", thickness=0.8, color=RULE, spaceBefore=2, spaceAfter=12),
            ]
        elif b.kind == "h2":
            story += [CondPageBreak(4 * cm), Paragraph(inline(b.text), styles["h2"])]
        elif b.kind == "h3":
            story += [CondPageBreak(3 * cm), Paragraph(inline(b.text), styles["h3"])]
        elif b.kind == "para":
            story.append(Paragraph(inline(b.text), styles["body"]))
        elif b.kind == "list":
            for k, item in enumerate(b.items, 1):
                story.append(Paragraph(inline(item), styles["bullet"], bulletText=f"{k}." if b.ordered else "•"))
            story.append(Spacer(1, 5))
        elif b.kind == "code":
            story += code_block(b.text, styles)
        elif b.kind == "math":
            story += formula(b.text, source)
        elif b.kind == "callout":
            story += callout(b, styles)
        elif b.kind == "table":
            story += table(b.items, styles, source)
        elif b.kind == "image":
            story += figure(b.text, b.lang, styles, source)
        elif b.kind == "pagebreak":
            story.append(PageBreak())
    return title, story


# ---------------------------------------------------------------------------
# Comprobaciones y montaje
# ---------------------------------------------------------------------------
def missing_glyphs(text: str) -> set[str]:
    """Caracteres del texto que no existen en DejaVu Sans (saldrían como cuadrados)."""
    face = pdfmetrics.getFont("DejaVu").face
    return {ch for ch in set(text) if ord(ch) >= 32 and ord(ch) not in face.charToGlyph}


def build(md_path: Path, styles) -> Path:
    source = md_path.name
    md = md_path.read_text(encoding="utf-8")
    prose = re.sub(r"\$\$.*?\$\$", "", md, flags=re.S)
    missing = missing_glyphs(prose)
    if missing:
        raise ValueError(f"{source}: caracteres sin glifo en DejaVu Sans: {' '.join(sorted(missing))}")

    title, story = to_flowables(parse(md, source), styles, source)
    out = DOCS / f"{md_path.stem}.pdf"

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 1.2 * cm, f"Quant-Backtester · {title}")
        canvas.drawRightString(PAGE_W - MARGIN, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.8 * cm, bottomMargin=2 * cm,
        title=title, author="Quant-Backtester", subject="Material de estudio",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return out


def main(prefixes: list[str]) -> None:
    register_fonts()
    styles = make_styles()
    sources = sorted(SOURCES.glob("*.md"))
    if prefixes:
        sources = [p for p in sources if p.name.startswith(tuple(prefixes))]
    if not sources:
        raise SystemExit("No hay fuentes que convertir en docs/fuentes/.")
    for path in sources:
        out = build(path, styles)
        print(f"{path.name:45} -> {out.relative_to(DOCS.parent)}")


if __name__ == "__main__":
    main(sys.argv[1:])
