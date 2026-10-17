# Publication figures

Each paper has a one-page `*-hero.pdf` cropped directly from its original PDF,
a self-contained `*-hero.svg` archive, and a 2400-pixel-wide `*-hero.png` web
preview rendered from the cropped PDF. The PDFs retain the original vector
artwork; any photographs retain their original raster data.

The website displays the full figure in its original aspect ratio and links
to the cropped PDF. PNG previews avoid browser rendering differences in complex
SVG masks, patterns, and outlined glyphs, while retaining the PDF's appearance.

To rebuild from a directory of original papers:

```powershell
python -m pip install "PyMuPDF==1.28.2"
python _scripts/export_publication_figures.py "path/to/original/papers"
```

`_data/publication_figures.json` records the source filename match, one-based
PDF page, and crop rectangle in PDF points measured from the top left. The
export only changes in-memory copies of the originals. Rendered QA images
are written to `tmp/pdfs/` for inspection.

When adjusting a crop, also update the matching `figure_summary`, `image_width`,
`image_height`, and `image_alt` in `_data/publications.yml`. The dimensions
equal the crop rectangle's width and height, keeping space reserved while
the preview loads. The PNG's higher pixel dimensions provide sharp previews on
high-density screens; the HTML dimensions continue to describe the crop ratio.
