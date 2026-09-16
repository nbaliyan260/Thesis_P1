"""Build the RECUT research artifacts without changing older studies."""
from pathlib import Path
import sys, json, hashlib, argparse
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[2]
import pdf_layout as layout
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Table, TableStyle
from pypdf import PdfReader

class Doc(BaseDocTemplate):
    def __init__(self, path, title):
        super().__init__(str(path), pagesize=layout.A4, leftMargin=layout.LEFT,
            rightMargin=layout.RIGHT, topMargin=layout.TOP, bottomMargin=layout.BOTTOM,
            title=title, author='Prepared for Nazish Baliyan; authorship pending review',
            subject='RECUT conditional research hypothesis and bounded prototype evidence')
        frame = Frame(layout.LEFT, layout.BOTTOM, layout.WIDTH,
            layout.H-layout.TOP-layout.BOTTOM, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='normal', frames=[frame], onPage=self.decorations)])
        self.outline_count, self.has_root = 0, False
    def decorations(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(layout.RULE)
        canvas.line(layout.LEFT, layout.H-34, layout.W-layout.RIGHT, layout.H-34)
        canvas.setFont('ArialB', 8)
        canvas.setFillColor(layout.ACCENT)
        canvas.drawString(layout.LEFT, layout.H-26, 'RECUT / DEPENDABLE LLM SYSTEMS')
        canvas.setFont('Arial', 8)
        canvas.setFillColor(layout.MUTED)
        canvas.drawRightString(layout.W-layout.RIGHT, layout.H-26, '16 SEPTEMBER 2026')
        canvas.line(layout.LEFT, 32, layout.W-layout.RIGHT, 32)
        canvas.drawString(layout.LEFT, 20, 'Supervisor-review material | Not a submission-readiness certificate')
        canvas.drawRightString(layout.W-layout.RIGHT, 20, str(doc.page))
        canvas.restoreState()
    afterFlowable = layout.DraftDoc.afterFlowable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = ROOT / 'output/pdf' / (source.stem + '.pdf')
    qa = ROOT / 'research-next/artifacts/qa' / source.stem
    qa.mkdir(parents=True, exist_ok=True)
    raw = source.read_text()
    assert raw.isascii(), 'Use ASCII typography in manuscript source'
    layout.SOURCE = source
    layout.widths = lambda headers: [layout.WIDTH*w for w in (
        [.40, .60] if source.stem == 'RECUT_04_Complete_Prototype_and_Validation' and headers == ['Model', 'Immutable revision'] else
        ([.29,.71] if len(headers)==2 else [.24,.38,.38]) if len(headers) in (2,3)
        else [1/len(headers)]*len(headers))]
    if source.stem == 'RECUT_04_Complete_Prototype_and_Validation':
        # Compact stage-3 numeric tables without changing earlier reports.
        # Keep 10-point prose and 9-point table text; trim vertical whitespace.
        layout.styles['body'].leading = 12.5
    story = layout.render_story(raw)
    if source.stem == 'RECUT_04_Complete_Prototype_and_Validation':
        for item in story:
            if isinstance(item, Table):
                item.setStyle(TableStyle([
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
    Doc(output, raw.splitlines()[0].lstrip('# ')).build(story)
    texts = [p.extract_text() for p in PdfReader(output).pages]
    assert all(len(t)>100 for t in texts)
    assert not any('ZZTOKEN' in t or '\ufffd' in t for t in texts)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pdf_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout_sha256=hashlib.sha256(Path(layout.__file__).read_bytes()).hexdigest(),
        pages=len(texts), page_text_lengths=list(map(len,texts)),
        visual_review='pending', output=str(output))
    (qa/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (qa/'text.txt').write_text('\n\n'.join(texts))
    print(json.dumps(metadata,indent=2))
if __name__ == '__main__': main()
