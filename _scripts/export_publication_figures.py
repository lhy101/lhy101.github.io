"""Export cropped PDF masters, SVG archives, and PNG web previews.

Requires PyMuPDF. Run from the repository root:
    python _scripts/export_publication_figures.py "path/to/source/papers"

Crop coordinates are PDF points, measured from the top left. Original PDF
content is placed directly into each new PDF, without screenshot rasterization.
PNG previews avoid browser differences in complex PDF-to-SVG masks and glyphs.
"""
import argparse
import json
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]


def export(source_dir, output_dir, qa_dir):
    specs = json.loads((ROOT / "_data/publication_figures.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    paths = list(source_dir.glob("*.pdf"))
    for stem, spec in specs.items():
        matches = [p for p in paths if spec["source_match"].casefold() in p.stem.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Expected one PDF for {stem}, found {len(matches)}")
        with fitz.open(matches[0]) as source:
            page_number = spec["page"] - 1
            clip = fitz.Rect(spec["clip"])
            original_page = source[page_number]
            bounds = original_page.rect
            if not bounds.contains(clip):
                raise ValueError(f"Crop outside source page: {stem}")
            # Remove unrelated page content in memory before importing. Otherwise
            # SVG exports carry invisible body text and images outside the crop.
            # Source files on disk are never changed; artwork stays as PDF paths.
            outside = [fitz.Rect(0, 0, bounds.width, clip.y0),
                       fitz.Rect(0, clip.y1, bounds.width, bounds.height),
                       fitz.Rect(0, clip.y0, clip.x0, clip.y1),
                       fitz.Rect(clip.x1, clip.y0, bounds.width, clip.y1)]
            for rect in outside:
                if not rect.is_empty:
                    original_page.add_redact_annot(rect, fill=False)
            # Keep images: a figure image can have a larger bounding box than
            # its visible pixels, so deleting intersecting images loses artwork.
            original_page.apply_redactions(images=0, graphics=1, text=0)
            with fitz.open() as crop:
                page = crop.new_page(width=clip.width, height=clip.height)
                page.show_pdf_page(page.rect, source, page_number, clip=clip)
                crop.set_metadata({"title": f"{spec['source_match']} - {spec['label']}",
                                   "subject": f"Figure crop from source PDF page {spec['page']}"})
                pdf_path = output_dir / f"{stem}-hero.pdf"
                crop.save(pdf_path, garbage=4, deflate=True)
        # Reopen the actual deliverable before generating the web derivative/QA.
        with fitz.open(pdf_path) as crop:
            assert len(crop) == 1
            page = crop[0]
            svg = page.get_svg_image(text_as_path=True)
            (output_dir / f"{stem}-hero.svg").write_text(svg, encoding="utf-8")
            scale = 2400 / page.rect.width
            preview = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            preview.save(output_dir / f"{stem}-hero.png")
            preview.save(qa_dir / f"{stem}-crop.png")
            print(f"{stem}: page {spec['page']}, {clip.width:g} x {clip.height:g} pt, {pdf_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/papers")
    parser.add_argument("--qa-dir", type=Path, default=ROOT / "tmp/pdfs")
    args = parser.parse_args()
    export(args.source_dir, args.output_dir, args.qa_dir)
