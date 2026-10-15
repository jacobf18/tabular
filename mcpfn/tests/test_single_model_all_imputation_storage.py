import inspect

import test_single_model_full_network_storage as previous_tests


SOURCE = inspect.getsource(previous_tests)
SOURCE = SOURCE.replace("import single_model_full_network_storage as diagnostics", "import single_model_all_imputation_storage as diagnostics")
SOURCE = SOURCE.replace('"checkpoint": 2304 * 1024**2, "root": 16 * 1024**2, "scratch": 32768 * 1024**2}',
    '"checkpoint": 2432 * 1024**2, "root": 16 * 1024**2, "scratch": 40 * 1024**3}')
SOURCE = SOURCE.replace('"checkpoint": 1536 * 1024**2, "root": 16 * 1024**2, "scratch": 16384 * 1024**2}',
    '"checkpoint": 2304 * 1024**2, "root": 16 * 1024**2, "scratch": 32768 * 1024**2}')
exec(compile(SOURCE, __file__, "exec"), globals())
