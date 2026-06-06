"""Genera PAPER.pdf desde PAPER.md (markdown → HTML con estilo académico → weasyprint).
Uso: python3 build_pdf.py [entrada.md] [salida.pdf]"""
import sys
import markdown
from weasyprint import HTML

src = sys.argv[1] if len(sys.argv) > 1 else "PAPER.md"
out = sys.argv[2] if len(sys.argv) > 2 else "PAPER.pdf"

md = open(src, encoding="utf-8").read()
# El primer '# ' es el título; el resto, cuerpo.
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])

CSS = """
@page { size: A4; margin: 2.2cm 2.0cm; @bottom-center { content: counter(page); font: 9pt Georgia; color:#666; } }
body { font: 10.5pt/1.45 Georgia, 'Times New Roman', serif; color:#111; }
h1 { font-size: 18pt; line-height:1.25; margin: 0 0 .1em; text-align:center; }
.authorblock { text-align:center; font-size:11pt; line-height:1.55; margin:.2em 0 1.1em; }
.authorblock code { background:none; font-size:10pt; }
h2 { font-size: 13pt; margin: 1.1em 0 .3em; border-bottom:1px solid #ccc; padding-bottom:2px; }
h3 { font-size: 11pt; margin: .9em 0 .2em; }
p, li { text-align: justify; }
table { border-collapse: collapse; width:100%; font-size: 9.2pt; margin: .6em 0; }
th, td { border:1px solid #bbb; padding:3px 6px; text-align:left; vertical-align:top; }
th { background:#f0f0f0; }
code { font-family:'DejaVu Sans Mono',monospace; font-size:8.8pt; background:#f4f4f4; padding:0 2px; }
pre { background:#f4f4f4; border:1px solid #ddd; padding:6px 8px; font-size:8.6pt; overflow-x:auto;
      white-space:pre-wrap; word-wrap:break-word; }
pre code { background:none; padding:0; }
blockquote { border-left:3px solid #888; margin:.6em 0; padding:.1em .9em; color:#333; background:#fafafa; }
a { color:#1a4f8b; text-decoration:none; }
hr { border:none; border-top:1px solid #ccc; margin:1.2em 0; }
em { color:#222; }
"""

html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
HTML(string=html).write_pdf(out)
print(f"→ {out}")
