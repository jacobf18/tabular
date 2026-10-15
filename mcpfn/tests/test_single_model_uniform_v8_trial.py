import inspect

import test_single_model_cohort_anchor_trial as previous_tests
from single_model_registered_adaptation import replace_once


SOURCE = inspect.getsource(previous_tests).replace('single_model_cohort_anchor_trial as trial',
    'single_model_uniform_v8_trial as trial')
SOURCE = SOURCE.replace('test_single_model_cohort_anchor_trial.py', 'test_single_model_uniform_v8_trial.py')
SOURCE = SOURCE.replace('every soft retention penalty increased', 'adverse13/26 probe penalty increases')
SOURCE = SOURCE.replace('"No probe"', '"probe/failed-trial warm start"')
SOURCE = SOURCE.replace('no additional inference model', 'No inference ensemble')
exec(compile(SOURCE, __file__, 'exec'), globals())
