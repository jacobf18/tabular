import os
from pathlib import Path
import subprocess
import sys


def test_interface_import_preserves_optional_baseline_dependencies():
    source = Path(__file__).resolve().parents[1] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source)}
    script = """
import importlib.abc
import sys

class BlockOptionalBaseline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'tabpfn_extensions':
            raise ImportError('Optional baseline deliberately unavailable')
        return None

sys.meta_path.insert(0, BlockOptionalBaseline())
from tabimpute.interface import ImputePFN
from tabimpute.routing import TwoExpertRouter
from tabimpute.tabimpute_v2 import TabImputeV2
assert 'tabimpute.prior.training_set_generation' not in sys.modules
assert 'tabimpute.prepreocess' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], env=environment, check=True)
