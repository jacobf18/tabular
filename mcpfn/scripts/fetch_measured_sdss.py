"""Download observed SDSS spectra and masks without using supplied PCA reconstructions."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


def fetch(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest['files']:
            assert hashlib.sha256((output / record['name']).read_bytes()).hexdigest() == record['sha256']
        return
    manifest = dict(created=datetime.now(timezone.utc).isoformat(), files=[],
                    usage='Only spectra, mask and wavelength coordinates may enter experiments; ignore supplied PCA coefficients, norms, reconstructions and labels.')
    for repo, relative in [('astroML/astroML-data', 'datasets/spec4000.npz'),
                           ('astroML/astroML', 'examples/datasets/compute_sdss_pca.py'),
                           ('astroML/astroML', 'astroML/datasets/sdss_corrected_spectra.py')]:
        response = requests.get(f'https://api.github.com/repos/{repo}/commits', params={'per_page': 1}, timeout=45)
        response.raise_for_status()
        revision = response.json()[0]['sha']
        url = f'https://raw.githubusercontent.com/{repo}/{revision}/{relative}'
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        name = Path(relative).name
        (output / name).write_bytes(response.content)
        manifest['files'].append(dict(name=name, url=url, revision=revision,
                                      sha256=hashlib.sha256(response.content).hexdigest(), bytes=len(response.content)))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Downloaded SDSS measured spectra and provenance to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    fetch(parser.parse_args().output)
