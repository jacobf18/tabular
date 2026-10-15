import inspect

import test_single_model_cohort_anchor_trial as previous_tests


SOURCE = inspect.getsource(previous_tests).replace('single_model_cohort_anchor_trial as trial',
    'single_model_family_balanced_trial as trial')
SOURCE = SOURCE.replace('test_single_model_cohort_anchor_trial.py', 'test_single_model_family_balanced_trial.py')
SOURCE = SOURCE.replace('every soft retention penalty increased', 'adverse25/26 retention-penalty increases')
exec(compile(SOURCE, __file__, 'exec'), globals())
