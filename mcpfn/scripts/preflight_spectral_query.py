"""Check real-anchor migration and exercise the spectral queue's trainer options."""
import argparse
import json
from pathlib import Path

import torch

from run_spectral_query_experiments import build_commands
from run_rowcol_controlled_experiments import run_logged, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid


def remove_option(command, flag):
    if flag not in command:
        return
    i = command.index(flag)
    j = i + 1
    while j < len(command) and not command[j].startswith("--"):
        j += 1
    del command[i:j]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, commands, _, anchor = build_commands(args.output)
    cfg = json.loads(commands["joint_low"][commands["joint_low"].index("--model-overrides-json") + 1])
    state = torch.load(anchor, map_location="cpu", weights_only=True)["model"]
    model = TabImputeModelRowColumnSVTHybrid(**cfg).eval()
    model.load_state_dict(state, strict=True)
    torch.manual_seed(927)
    source = torch.randn(1, 20, 16)
    source[:, ::3, ::2] = float("nan")
    mask = source.isnan()
    with torch.no_grad():
        reference = model(source, decode_mask=mask)
    del model
    model = TabImputeModelRowColumnSVTHybrid(**{**cfg, "cell_query_axis_balanced": True}).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert len(missing) == 6 and all("axis_gate" in n or "axis_blend" in n for n in missing)
    with torch.no_grad():
        actual = model(source, decode_mask=mask)
    torch.testing.assert_close(reference, actual, rtol=0, atol=0)
    write_json(args.output / "migration.json", dict(anchor=str(anchor), missing=missing,
        max_abs_error=float((actual - reference).abs().max()), precision="float32_cpu"))
    del model, state
    tiny = dict(embedding_size=32, mlp_hidden_size=48, num_attention_heads=4, num_layers=1,
        extra_decoder_hidden_size=64, count_hidden_size=8, rabel_exponent_embedding_size=8,
        rabel_num_kernels=8, observed_attention_dim=8, observed_attention_hidden_size=16,
        svt_feature_decoder_hidden_size=32, svt_init_rank=3, svt_max_rank=5,
        svt_mixture_hidden_size=32, svt_mixture_steps=[0, 1, 2], svt_mixture_cv_min_count=1,
        cell_query_width=16, cell_query_chunk_size=8)
    for tag in ("axis_low", "frozen_low"):
        command = list(commands[tag])
        for flag in ("--init-model-checkpoint", "--allow-partial-init", "--fixed-validation-cache", "--snapshot-steps"):
            remove_option(command, flag)
        model_cfg = json.loads(command[command.index("--model-overrides-json") + 1])
        for flag, value in {"--run-name": f"smoke_spectral_{tag}",
                "--model-overrides-json": json.dumps({**model_cfg, **tiny}),
                "--phases-json": json.dumps([dict(name="smoke", end_step=6, rows_low=20,
                    rows_high=20, cols_low=16, cols_high=16, batch_size=2)]),
                "--warmup-steps": 1, "--decay-steps": 5, "--checkpoint-every": 6,
                "--metrics-jsonl-every": 1, "--update-diagnostics-every": 1}.items():
            replace_arg(command, flag, value)
        command += ["--work-dir", str(args.output / tag)]
        run_logged(command, args.output / f"{tag}.log")
    write_json(args.output / "complete.json", {"migration": "exact", "trainer_smokes": ["axis_low", "frozen_low"]})


if __name__ == "__main__":
    main()
