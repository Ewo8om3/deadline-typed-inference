# Building the paper

## Requirements

- Pandoc
- A Chromium-based browser for PDF rendering

## Web edition

From the repository root:

```bash
pandoc paper/deadline-typed-inference-design.md \
  --standalone \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --embed-resources \
  --resource-path=paper \
  --css=paper/design.css \
  -o docs/index.html
```

## PDF

After generating `docs/index.html`, print it with a headless Chromium build. On macOS with Google Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$PWD/paper/deadline-typed-inference-design.pdf" \
  "file://$PWD/docs/index.html"
```

The committed artifact is US Letter, 58 pages in v1.0. Page count may change when content changes.

## Verification

Recommended checks:

```bash
pdfinfo paper/deadline-typed-inference-design.pdf
pdftotext -layout paper/deadline-typed-inference-design.pdf - | wc -w
rg -i "TODO|TBD|FIXME|lorem ipsum" paper docs
```

Render a contact sheet or inspect representative pages before publishing a new PDF.
