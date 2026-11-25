"""Stack two representation figures without rasterizing their vector content."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

import fitz
import numpy as np


LABEL_GUTTER = 16.
PANEL_GAP = 12.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_clip(page, top_panel):
    pixmap = page.get_pixmap(dpi=144, alpha=False)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    occupied = np.flatnonzero((pixels[:, :, :3] < 255).any(axis=(1, 2)))
    rectangle = fitz.Rect(page.rect)
    if len(occupied):
        scale = page.rect.height / pixmap.height
        if top_panel:
            rectangle.y1 = min(rectangle.y1, (occupied[-1] + 1) * scale + 2.)
        else:
            rectangle.y0 = max(rectangle.y0, occupied[0] * scale - 2.)
    return rectangle


def assemble(latent, groups, output, latent_generator=None, *, width=None, panel_contents=None, title=None,
             gap=PANEL_GAP, trim_inner=False):
    inputs = [Path(latent), Path(groups)]
    output = Path(output)
    targets = [output.with_suffix(extension) for extension in ('.pdf', '.png', '.json')]
    if {path.resolve() for path in inputs} & {path.resolve() for path in targets}:
        raise ValueError('Output must not overwrite a source figure')
    if not math.isfinite(gap) or gap < 0:
        raise ValueError('Panel gap must be nonnegative and finite')
    sources = {str(path.resolve()): digest(path) for path in inputs}
    generator = None
    if latent_generator is not None:
        generator = dict(path=str(Path(latent_generator).resolve()), sha256=digest(latent_generator))
    with ExitStack() as stack:
        documents = [stack.enter_context(fitz.open(path)) for path in inputs]
        if any(len(document) != 1 for document in documents):
            raise ValueError('Each source must contain exactly one page')
        widths = [document[0].rect.width for document in documents]
        if width is None and not math.isclose(widths[0], widths[1], abs_tol=.01):
            raise ValueError('Source widths must match to preserve the original scale')
        target_width = widths[0] if width is None else float(width)
        if not math.isfinite(target_width) or target_width <= 0:
            raise ValueError('Output width must be positive and finite')
        clips = [inner_clip(document[0], index == 0) if trim_inner else document[0].rect
                 for index, document in enumerate(documents)]
        heights = [clip.height * target_width / source_width for clip, source_width in zip(clips, widths)]
        combined = stack.enter_context(fitz.open())
        page = combined.new_page(width=LABEL_GUTTER + target_width, height=sum(heights) + gap)
        top = 0.
        placements = []
        for label, document, height, clip in zip(('a.', 'b.'), documents, heights, clips):
            rectangle = fitz.Rect(LABEL_GUTTER, top, LABEL_GUTTER + target_width, top + height)
            page.show_pdf_page(rectangle, document, 0, clip=clip)
            page.insert_text((1., top + 10.), label, fontsize=10, fontname='tibo')
            placements.append(dict(label=label, source_rectangle=list(clip),
                                   original_source_rectangle=list(document[0].rect),
                                   output_rectangle=list(rectangle)))
            top += height + gap
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.set_metadata({'title': title or 'Row representations: latent factors and latent groups',
                               'creator': Path(__file__).name})
        combined.save(targets[0], garbage=4, deflate=True)
        page.get_pixmap(dpi=300, alpha=False).save(targets[1])
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f'Source changed during assembly: {path}')
    contents = panel_contents or ('Latent row factors decoded from row-CLS tokens',
                                  'Latent row groups in mean-pooled cell embeddings')
    manifest = dict(panels=[dict(label=label, content=content)
                            for label, content in zip(('a.', 'b.'), contents)],
                    source_sha256=sources, latent_generator=generator, script_sha256=digest(__file__),
                    panel_width_points=target_width,
                    gap_points=gap, trim_inner_blank_margins=trim_inner,
                    placements=placements,
                    composition='Vector-preserving; only inner blank margins trimmed when requested; no refitting or rescoring',
                    artifacts_sha256={str(path.resolve()): digest(path) for path in targets[:2]})
    targets[2].write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {targets[0]} and {targets[1]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent', '--top', type=Path, required=True)
    parser.add_argument('--groups', '--bottom', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output path stem, without extension')
    parser.add_argument('--latent-generator', type=Path, help='Record the original generator path and hash')
    parser.add_argument('--width', type=float, help='Uniform panel width in PDF points; preserves aspect ratios')
    parser.add_argument('--top-content', default='Latent row factors decoded from row-CLS tokens')
    parser.add_argument('--bottom-content', default='Latent row groups in mean-pooled cell embeddings')
    parser.add_argument('--title', help='PDF document title')
    parser.add_argument('--gap', type=float, default=PANEL_GAP, help='Vertical panel gap in PDF points')
    parser.add_argument('--trim-inner', action='store_true', help='Trim blank space at the inner panel edges')
    args = parser.parse_args()
    assemble(args.latent, args.groups, args.output, args.latent_generator, width=args.width,
             panel_contents=(args.top_content, args.bottom_content), title=args.title,
             gap=args.gap, trim_inner=args.trim_inner)


if __name__ == '__main__':
    main()
