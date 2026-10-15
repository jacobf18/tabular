"""Use the original general teacher for all synthetic imputation training only."""

import hashlib
import inspect

import single_model_structured_teacher as template
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY = {"legacy": "general", "structured": "general", "recsys": "specialist"}
ADAPTED_SOURCE = replace_assignment(inspect.getsource(template), "POLICY", POLICY)
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE,
    '"""Training-only structured-general distillation with checkpoint-bound teacher identities."""',
    '"""Training-only general-imputation distillation; original validation references stay unchanged."""')
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
