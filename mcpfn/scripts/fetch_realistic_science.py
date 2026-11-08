"""Download measured environmental data and retain authoritative provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


SOURCES = {
    'air_quality': (360, 'air+quality.zip'),
    'beijing': (501, 'beijing+multi+site+air+quality+data.zip'),
}


def fetch(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {'created': datetime.now(timezone.utc).isoformat(), 'files': []}
    for name, (identifier, filename) in SOURCES.items():
        destination = output / f'{name}.zip'
        url = f'https://archive.ics.uci.edu/static/public/{identifier}/{filename}'
        if not destination.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            destination.write_bytes(response.content)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        previous = next((record for record in manifest['files'] if record['name'] == name), None)
        if previous is not None:
            assert previous['sha256'] == digest
        else:
            metadata_url = f'https://archive.ics.uci.edu/api/dataset?id={identifier}'
            response = requests.get(metadata_url, timeout=60)
            response.raise_for_status()
            (output / f'{name}_metadata.json').write_text(json.dumps(response.json(), indent=2))
            manifest['files'].append(dict(name=name, url=url, metadata_url=metadata_url,
                                          sha256=digest, bytes=destination.stat().st_size))
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(name, destination, destination.stat().st_size, digest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    fetch(parser.parse_args().output)
