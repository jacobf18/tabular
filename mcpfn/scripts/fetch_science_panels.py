"""Download source archives for additional measured scientific panels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse

import requests


ROOT = Path('/mnt/matrixformer_scratch/science_panels_20261106')


def sources():
    result = {
        'water_treatment.zip': 'https://archive.ics.uci.edu/static/public/106/water+treatment+plant.zip',
        'intel_description.html': 'https://db.csail.mit.edu/labdata/labdata.html',
        'intel_data.txt.gz': 'https://db.csail.mit.edu/labdata/data.txt.gz',
        'intel_locations.txt': 'https://db.csail.mit.edu/labdata/mote_locs.txt',
    }
    for site in ('01646500', '01579550'):
        parameters = dict(format='json', sites=site, startDT='2019-01-01', endDT='2019-12-31',
                          parameterCd='00010,00060,00095,00300,00400,63680,99133', siteStatus='all')
        result[f'usgs_{site}_2019.json'] = 'https://waterservices.usgs.gov/nwis/iv/?' + urllib.parse.urlencode(parameters)
    result['GLODAPv2.2023_Atlantic_Ocean.mat'] = 'https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/GLODAPv2.2023_Atlantic_Ocean.mat'
    revision = 'a5b1efce24fcffc8b9e6dd3a0d1e54fa384a01d5'
    for name in ('CANYONB.R', 'wgts_AT.txt', 'README.md', 'LICENSE'):
        result['canyon_b/' + name] = f'https://raw.githubusercontent.com/HCBScienceProducts/CANYON-B/{revision}/{name}'
    return result


def fetch(item, output):
    name, url = item
    destination = output / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = dict(url=url, destination=str(destination))
    try:
        if not destination.exists():
            response = requests.get(url, timeout=(30, 240))
            response.raise_for_status()
            temporary = destination.with_suffix(destination.suffix + '.partial')
            temporary.write_bytes(response.content)
            temporary.rename(destination)
        record.update(status='downloaded', bytes=destination.stat().st_size,
                      sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    except Exception as error:
        record.update(status='failed', error=str(error))
    print(name, record['status'], record.get('bytes', record.get('error')), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: fetch(item, arguments.output), sources().items()))
    (arguments.output / 'manifest.json').write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).isoformat(), sources=records), indent=2) + '\n')
