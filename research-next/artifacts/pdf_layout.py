"""Portable Markdown-to-ReportLab layout for the RECUT research artifacts.

This module contains no experiment checks, output paths, file writes, or build
entry point. Callers may set SOURCE (for relative image paths) and override
widths before calling render_story. Importing only registers the layout fonts.
"""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Table, TableStyle,
)


SOURCE = Path(__file__).with_name('document.md')
FONT_DIR = Path('/System/Library/Fonts/Supplemental')


def register_fonts(font_dir=FONT_DIR):
    """Keep macOS Arial/Courier New when present; otherwise use built-in fonts.

    The stable aliases also serve callers' page decorations. The fallback uses
    ReportLab's PDF base fonts and needs no platform-specific font files.
    """
    selected = {}
    for name, filename, fallback in [
        ('Arial', 'Arial.ttf', 'Helvetica'),
        ('ArialB', 'Arial Bold.ttf', 'Helvetica-Bold'),
        ('ArialI', 'Arial Italic.ttf', 'Helvetica-Oblique'),
        ('Mono', 'Courier New.ttf', 'Courier'),
    ]:
        path = Path(font_dir) / filename
        if path.is_file():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            selected[name] = str(path)
        else:
            pdfmetrics.registerFont(pdfmetrics.Font(name, fallback, 'WinAnsiEncoding'))
            selected[name] = fallback
    pdfmetrics.registerFontFamily(
        'Arial', normal='Arial', bold='ArialB', italic='ArialI', boldItalic='ArialB')
    return selected


FONT_BACKENDS = register_fonts()

INK = colors.HexColor('#17283B')
ACCENT = colors.HexColor('#12657A')
MUTED = colors.HexColor('#536174')
LIGHT = colors.HexColor('#EEF4F7')
RULE = colors.HexColor('#CCD8E0')
W, H = A4
LEFT, RIGHT, TOP, BOTTOM = 48, 48, 52, 43
WIDTH = W - LEFT - RIGHT

styles = {
    'body': ParagraphStyle('body', fontName='Arial', fontSize=10, leading=14.2,
                           textColor=INK, spaceAfter=8, allowWidows=0, allowOrphans=0),
    'h1': ParagraphStyle('h1', fontName='ArialB', fontSize=17, leading=21,
                         textColor=INK, spaceBefore=3, spaceAfter=12, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='ArialB', fontSize=13, leading=17,
                         textColor=ACCENT, spaceBefore=6, spaceAfter=8, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='ArialB', fontSize=10.7, leading=14,
                         textColor=INK, spaceBefore=7, spaceAfter=5, keepWithNext=True),
    'cell': ParagraphStyle('cell', fontName='Arial', fontSize=9, leading=12,
                           textColor=INK, spaceAfter=0),
    'th': ParagraphStyle('th', fontName='ArialB', fontSize=9, leading=12,
                         textColor=colors.white),
    'quote': ParagraphStyle('quote', fontName='ArialI', fontSize=10, leading=14.5,
                            textColor=INK, leftIndent=12, rightIndent=10,
                            borderColor=ACCENT, borderWidth=1, borderPadding=9,
                            backColor=LIGHT, spaceBefore=17, spaceAfter=12),
    'list': ParagraphStyle('list', fontName='Arial', fontSize=10, leading=14.2,
                           textColor=INK, leftIndent=15, firstLineIndent=-15,
                           spaceAfter=8, allowWidows=0, allowOrphans=0),
}


def inline(text):
    saved = []

    def stash(value):
        saved.append(value)
        return f'ZZTOKEN{len(saved)-1}ZZ'

    text = re.sub(r'`([^`]+)`', lambda m: stash('<font name="Mono" size="8.2">' + html.escape(m.group(1)) + '</font>'), text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: stash('<link href="' + html.escape(m.group(2), quote=True) + '" color="#12657A"><u>' + html.escape(m.group(1)) + '</u></link>'), text)
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    for index, value in enumerate(saved):
        text = text.replace(f'ZZTOKEN{index}ZZ', value)
    return text


def widths(headers):
    """Default table-width heuristics; the RECUT builder overrides this policy."""
    count = len(headers)
    if count == 2:
        weights = [.36, .64]
        if headers[0] == 'Record':
            weights = [.52, .48]
        elif headers[0] == 'Requirement':
            weights = [.58, .42]
        elif headers[0] == 'Question':
            weights = [.43, .57]
    elif count == 3:
        weights = [.25, .20, .55]
        if headers[0] == 'Venue': weights = [.13, .15, .72]
        elif headers[0] == 'Kernel': weights = [.12, .48, .40]
        elif headers[0].startswith('Protection'): weights = [.37, .33, .30]
        elif headers[0] == 'Outcome': weights = [.23, .52, .25]
        elif headers[0] == 'Stage': weights = [.28, .15, .57]
    elif count == 4: weights = [.12, .15, .49, .24]
    elif count == 5: weights = [.22, .18, .13, .19, .28]
    elif count == 6: weights = [.14, .18, .13, .11, .14, .30]
    else: weights = [1 / count] * count
    return [WIDTH * weight for weight in weights]


def make_table(lines):
    raw = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    raw = [row for row in raw if not all(re.fullmatch(r':?-+:?', cell) for cell in row)]
    rows = [[Paragraph(inline(cell), styles['th' if row_index == 0 else 'cell']) for cell in row]
            for row_index, row in enumerate(raw)]
    table = Table(rows, colWidths=widths(raw[0]), repeatRows=1, hAlign='LEFT', spaceBefore=3, spaceAfter=9)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, -1), (-1, -1), .5, RULE),
    ]))
    return table


def render_story(source):
    story = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == '<!-- PAGE -->':
            story.append(PageBreak())
            index += 1
            continue
        image_match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if image_match:
            path = (SOURCE.parent / image_match.group(2)).resolve()
            item = Image(str(path))
            item.drawHeight = item.imageHeight / item.imageWidth * WIDTH
            item.drawWidth = WIDTH
            item.spaceAfter = 8
            story.append(item)
            index += 1
            continue
        if line.startswith('|'):
            block = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                block.append(lines[index])
                index += 1
            story.append(make_table(block))
            continue
        match = re.match(r'^(#{1,3}) (.*)', line)
        if match:
            paragraph = Paragraph(inline(match.group(2)), styles['h' + str(len(match.group(1)))])
            paragraph.outline_label = match.group(2)
            paragraph.outline_level = 0 if len(match.group(1)) == 1 else 1
            story.append(paragraph)
            index += 1
            continue
        block = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r'^(#|\||<!--|!\[|\d+\. )', lines[index]):
            block.append(lines[index].strip())
            index += 1
        text = ' '.join(block)
        style = 'body'
        if text.startswith('> '):
            text, style = text[2:], 'quote'
        elif re.match(r'^\d+\. ', text):
            style = 'list'
        story.append(Paragraph(inline(text), styles[style]))
    return story


class DraftDoc(BaseDocTemplate):
    """Neutral layout document; project builders may borrow afterFlowable only."""

    def __init__(self, path, title='Research working draft'):
        super().__init__(str(path), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                         topMargin=TOP, bottomMargin=BOTTOM, title=title)
        frame = Frame(LEFT, BOTTOM, WIDTH, H-TOP-BOTTOM, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='normal', frames=[frame])])
        self.outline_count = 0
        self.has_root = False

    def afterFlowable(self, flowable):
        if hasattr(flowable, 'outline_label'):
            key = f'section-{self.outline_count}'
            self.outline_count += 1
            level = flowable.outline_level if self.has_root else 0
            self.has_root = True
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.outline_label, key, level, closed=False)
