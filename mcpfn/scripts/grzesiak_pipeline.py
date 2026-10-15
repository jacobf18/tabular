"""Durable continuation of the declared campaign, with audited paper refreshes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from grzesiak_benchmark import REPO, WORK, write_json


PAPER = Path('/root/MatrixFormer')


def active(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return 'grzesiak_evaluate.py' in Path(f'/proc/{pid}/cmdline').read_text()
    except ProcessLookupError:
        return False
    except FileNotFoundError:
        return False


def run(command, work, log_name):
    with (work / log_name).open('a') as log:
        log.write('\nCOMMAND ' + repr(command) + '\n')
        log.flush()
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=os.environ | {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                              'PYTHONPATH': str(REPO / 'scripts') + ':' + str(REPO / 'src')})


def refresh(work, final=False):
    for command in [[sys.executable, str(REPO / 'scripts/grzesiak_summarize.py'), '--work', str(work)],
                    [sys.executable, str(REPO / 'scripts/grzesiak_audit.py'), '--work', str(work)],
                    [sys.executable, str(PAPER / 'scripts/export_grzesiak_results.py'), '--source', str(work)]]:
        process = run(command, work, 'refresh.log')
        if process.wait() != 0:
            raise RuntimeError(f'Refresh failed: {command}')
    if final:
        directory = Path('/tmp/matrixformer-latex')
        directory.mkdir(exist_ok=True)
        with (work / 'latex.log').open('a') as log:
            subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                            '-outdir=' + str(directory), 'main.tex'], cwd=PAPER, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--existing-broad-pid', type=int)
    parser.add_argument('--interval', type=int, default=300)
    arguments = parser.parse_args()
    work = arguments.work
    lock = (work / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = json.loads((work / 'campaign.json').read_text())
    state = {'state': 'running', 'supervisor_pid': os.getpid(), 'started_unix': time.time(),
             'broad_pid': arguments.existing_broad_pid, 'fresh_pid': None}
    write_json(work / 'campaign_status.json', state)
    if shutil.disk_usage(work).free < 2 * 1024 ** 3:
        raise RuntimeError('Need at least 2 GiB free before starting the expanded campaign')
    command = [sys.executable, str(REPO / 'scripts/grzesiak_evaluate.py'), '--work', str(work),
               '--scope', 'fresh', '--workers', '6', '--energy', '--methods', *campaign['fresh_methods']]
    fresh = run(command, work, 'fresh-full.log')
    state['fresh_pid'] = fresh.pid
    broad = None
    broad_verified = False
    try:
        while True:
            if not active(arguments.existing_broad_pid) and broad is None and not broad_verified:
                broad = run([sys.executable, str(REPO / 'scripts/grzesiak_evaluate.py'), '--work', str(work),
                             '--scope', 'broad', '--workers', '12', '--timeout', '600', '--methods', *campaign['broad_methods']],
                            work, 'broad-resume.log')
                state['broad_pid'] = broad.pid
            if broad is not None and broad.poll() is not None:
                if broad.returncode:
                    raise RuntimeError(f'Broad continuation failed: {broad.returncode}')
                broad_verified = True
                broad = None
            if fresh.poll() is not None and fresh.returncode:
                raise RuntimeError(f'Fresh campaign failed: {fresh.returncode}')
            state['updated_unix'] = time.time()
            state['disk_free_bytes'] = shutil.disk_usage(work).free
            write_json(work / 'campaign_status.json', state)
            refresh(work)
            if fresh.poll() == 0 and broad_verified:
                refresh(work, final=True)
                state['state'] = 'finished_all_requested_attempts'
                state['finished_unix'] = time.time()
                write_json(work / 'campaign_status.json', state)
                refresh(work)
                break
            time.sleep(arguments.interval)
    except Exception as error:
        state.update(state='supervision_error', error=str(error), updated_unix=time.time())
        write_json(work / 'campaign_status.json', state)
        raise


if __name__ == '__main__':
    main()
