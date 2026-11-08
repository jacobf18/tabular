"""Fetch version-pinned, publicly distributed measured spectroscopy datasets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


FILES = {
    'chstan/pyarpes-datasets': ('README.md', 'data/photon_energy.nc', 'data/temperature_dependence.nc',
                             'data/fermi_surface.nc', 'data/nano_xps.nc'),
    'gokriznastic/HybridSN': ('README.md', 'LICENSE', 'data/Indian_pines_corrected.mat',
                           'data/Salinas_corrected.mat', 'data/PaviaU.mat'),
}


def fetch(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        for record in manifest['files']:
            path = output / record['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        print('Existing downloads verified')
        return
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), files=[], revisions={},
                    note='Data only, not imported third-party code. PyARPES examples may be binned; these are not newly simulated matrices. '
                         'HybridSN hosts standard AVIRIS/ROSIS measured hyperspectral scenes. No class labels are downloaded or used.')
    for repo, paths in FILES.items():
        response = requests.get(f'https://api.github.com/repos/{repo}/commits', params={'per_page': 1}, timeout=60)
        response.raise_for_status()
        revision = response.json()[0]['sha']
        manifest['revisions'][repo] = revision
        for relative in paths:
            url = f'https://raw.githubusercontent.com/{repo}/{revision}/{relative}'
            destination = output / repo.split('/')[-1] / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            destination.write_bytes(response.content)
            manifest['files'].append(dict(repo=repo, url=url, path=str(destination.relative_to(output)),
                                          bytes=destination.stat().st_size,
                                          sha256=hashlib.sha256(response.content).hexdigest()))
            print(f'{repo}/{relative}: {destination.stat().st_size} bytes', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    fetch(parser.parse_args().output)
