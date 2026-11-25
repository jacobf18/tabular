import json

import fitz
import pytest

from assemble_row_representation_panels import LABEL_GUTTER, PANEL_GAP, assemble, digest


def source_pdf(path, text, height=100, width=396, pages=1):
    with fitz.open() as document:
        for index in range(pages):
            page = document.new_page(width=width, height=height)
            page.insert_text((30, 20), text)
            page.draw_line((30, 50), (350, 70), color=(0, 0, 1))
        document.save(path)
    return path


def test_assembly_preserves_vectors_order_and_source_files(tmp_path):
    latent = source_pdf(tmp_path / 'latent.pdf', 'Latent row factors', height=100)
    groups = source_pdf(tmp_path / 'groups.pdf', 'Row-group embeddings', height=110)
    output = tmp_path / 'combined'
    hashes = {str(path): digest(path) for path in (latent, groups)}
    assemble(latent, groups, output)
    with fitz.open(output.with_suffix('.pdf')) as document:
        assert len(document) == 1
        page = document[0]
        assert page.rect.width == 396 + LABEL_GUTTER
        assert page.rect.height == 210 + PANEL_GAP
        assert not page.get_images()
        assert len(page.get_drawings()) == 2
        assert page.search_for('Latent row factors')[0].y0 < page.search_for('Row-group embeddings')[0].y0
        spans = [span for block in page.get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']]
        labels = [span for span in spans if span['text'] in ('a.', 'b.')]
        assert [span['text'] for span in labels] == ['a.', 'b.']
        assert all('Bold' in span['font'] for span in labels)
        for source, top in ((latent, 0), (groups, 100 + PANEL_GAP)):
            with fitz.open(source) as original:
                rectangle = original[0].rect + (LABEL_GUTTER, top, LABEL_GUTTER, top)
                assert page.get_pixmap(clip=rectangle).samples == original[0].get_pixmap().samples
    manifest = json.loads(output.with_suffix('.json').read_text())
    assert manifest['source_sha256'] == hashes
    assert {path: digest(path) for path in hashes} == hashes
    for path, expected in manifest['artifacts_sha256'].items():
        assert digest(path) == expected


@pytest.mark.parametrize('problem', ['pages', 'width', 'overwrite'])
def test_invalid_inputs_are_rejected(tmp_path, problem):
    latent = source_pdf(tmp_path / 'latent.pdf', 'Latent row factors')
    groups = source_pdf(tmp_path / 'groups.pdf', 'Row groups',
                        width=400 if problem == 'width' else 396,
                        pages=2 if problem == 'pages' else 1)
    output = latent.with_suffix('') if problem == 'overwrite' else tmp_path / 'combined'
    with pytest.raises(ValueError):
        assemble(latent, groups, output)


def test_different_widths_can_be_uniformly_scaled_and_labeled(tmp_path):
    matrix = source_pdf(tmp_path / 'matrix.pdf', 'Row-only matrix', width=518.4, height=190.8)
    embeddings = source_pdf(tmp_path / 'embeddings.pdf', 'Row embeddings', width=396, height=97.2)
    output = tmp_path / 'row_group_panels'
    hashes = {str(path): digest(path) for path in (matrix, embeddings)}
    assemble(matrix, embeddings, output, width=396,
             panel_contents=('Matrix construction', 'Row embedding recovery'), title='Row-group example')
    with fitz.open(output.with_suffix('.pdf')) as document:
        page = document[0]
        assert page.rect.width == 396 + LABEL_GUTTER
        assert page.rect.height == pytest.approx(190.8 * 396 / 518.4 + 97.2 + PANEL_GAP)
        assert len(page.get_drawings()) == 2
        assert not page.get_images()
        assert document.metadata['title'] == 'Row-group example'
        assert page.search_for('Row-only matrix')[0].y0 < page.search_for('Row embeddings')[0].y0
    manifest = json.loads(output.with_suffix('.json').read_text())
    assert manifest['panels'] == [{'label': 'a.', 'content': 'Matrix construction'},
                                  {'label': 'b.', 'content': 'Row embedding recovery'}]
    for placement in manifest['placements']:
        source = fitz.Rect(placement['source_rectangle'])
        target = fitz.Rect(placement['output_rectangle'])
        assert target.width == pytest.approx(396)
        assert target.height / target.width == pytest.approx(source.height / source.width)
    assert {path: digest(path) for path in hashes} == hashes


@pytest.mark.parametrize('width', [0, -1, float('inf'), float('nan')])
def test_invalid_width_is_rejected(tmp_path, width):
    top = source_pdf(tmp_path / 'top.pdf', 'Matrix')
    bottom = source_pdf(tmp_path / 'bottom.pdf', 'Embeddings')
    with pytest.raises(ValueError, match='positive and finite'):
        assemble(top, bottom, tmp_path / 'combined', width=width)


def test_inner_whitespace_can_be_removed_without_rasterizing_content(tmp_path):
    top = source_pdf(tmp_path / 'top.pdf', 'Matrix construction', height=150)
    bottom = source_pdf(tmp_path / 'bottom.pdf', 'Embeddings', height=150)
    output = tmp_path / 'compact'
    assemble(top, bottom, output, gap=2, trim_inner=True)
    with fitz.open(output.with_suffix('.pdf')) as document:
        page = document[0]
        assert page.rect.height < 240
        assert page.search_for('Matrix construction')
        assert page.search_for('Embeddings')
        assert len(page.get_drawings()) == 2
        assert not page.get_images()
    manifest = json.loads(output.with_suffix('.json').read_text())
    assert manifest['gap_points'] == 2
    assert manifest['trim_inner_blank_margins']
    top_placement, bottom_placement = manifest['placements']
    assert top_placement['source_rectangle'][3] < 150
    assert bottom_placement['source_rectangle'][1] > 0
    assert bottom_placement['output_rectangle'][1] - top_placement['output_rectangle'][3] == pytest.approx(2)
