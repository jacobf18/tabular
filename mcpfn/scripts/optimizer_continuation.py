"""Retain AdamW history while explicitly restarting only the learning-rate schedule."""


def retain_adamw_state(optimizer, checkpoint, model):
    groups = checkpoint["optimizer"]["param_groups"]
    if len(groups) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError("Moment continuation currently requires one parameter group")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if names != [name for name in checkpoint["model"] if name in set(names)]:
        raise ValueError("Source model parameter order differs from destination")
    parameters = optimizer.param_groups[0]["params"]
    ids = groups[0]["params"]
    if len(ids) != len(parameters) or len(names) != len(parameters):
        raise ValueError("Source optimizer parameter count differs from destination")
    for key in ("betas", "eps", "weight_decay", "amsgrad"):
        if groups[0].get(key) != optimizer.param_groups[0].get(key):
            raise ValueError(f"Optimizer {key} changed; not a moments-only comparison")
    state = checkpoint["optimizer"]["state"]
    if not state:
        raise ValueError("Source optimizer has no accumulated state")
    for parameter, identifier in zip(parameters, ids):
        for key in ("exp_avg", "exp_avg_sq"):
            if identifier in state and state[identifier][key].shape != parameter.shape:
                raise ValueError(f"Source {key} shape differs from destination")
    # LambdaLR has already initialized the new schedule. Preserve all new group
    # options while loading only the accumulated moments and per-parameter steps.
    options = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer.param_groups[0].update(options)
    return {"source_epoch": int(checkpoint["epoch"]), "state_count": len(state),
            "schedule_restarted": True, "optimizer_steps_retained": True}
