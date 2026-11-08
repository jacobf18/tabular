"""Fixed adapters and explicit scoring for the five-seed imputation campaign."""

import json
from pathlib import Path
import random
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd

from evaluate_dwip_missbench_uci import score_predictions


REPO = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth')
SCIENTIFIC = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2/snapshots/step_3000/checkpoint.pth')
MODEL_PATHS = {'matrixformer': ORIGINAL, 'matrixformer_science_step3000': SCIENTIFIC}
TABIMPUTE = Path('/root/checkpoints/masters/mcar/step-117000.ckpt')
CPU_METHODS = ('column_mean', 'knn', 'missforest', 'ice', 'softimpute', 'mice', 'hyperimpute', 'forestdiffusion')
GPU_METHODS = ('matrixformer', 'matrixformer_science_step3000', 'tabimpute', 'gain', 'miwae',
               'ot_sinkhorn', 'remasker', 'cacti', 'diffputer')
R_METHODS = ('r_mice_pmm', 'r_mice_cart', 'r_mice_rf', 'r_mice_drf', 'r_mice_norm',
             'r_mice_norm_predict', 'r_mice_norm_nob', 'r_mice_midastouch', 'r_mice_boot',
             'r_missforest', 'r_missmda_pca_em', 'r_missmda_pca_reg',
             'r_missmda_MIPCA_em', 'r_missmda_MIPCA_reg')
LABELS = dict(matrixformer='MatrixFormer', matrixformer_science_step3000='MatrixFormer (science step 3000)',
              tabimpute='TabImpute', column_mean='Column mean', knn='K-NN', missforest='MissForest',
              ice='ICE', softimpute='SoftImpute', mice='MICE-Forest', hyperimpute='HyperImpute',
              forestdiffusion='ForestDiffusion', gain='GAIN', miwae='MIWAE', ot_sinkhorn='OT (Sinkhorn)',
              remasker='ReMasker', cacti='CMAE (no context)', diffputer='DiffPuter')
LABELS.update({name: name.removeprefix('r_') + ' (R)' for name in R_METHODS})


def observed_standardize(missing):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        center = np.nanmean(missing, axis=0)
        scale = np.nanstd(missing, axis=0)
    center = np.where(np.isfinite(center), center, 0.)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.)
    return (missing - center) / scale, center, scale


def metrics(truth, missing, prediction, suite):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError('Require finite predictions for every cell')
    observed = np.isfinite(missing)
    if not np.array_equal(prediction[observed], missing[observed]):
        raise ValueError('Observed values changed')
    result = score_predictions(truth, prediction, missing, suite)
    selected = ~observed
    scored_columns = int(selected.any(axis=0).sum()) - result['nrmse_skipped_constant_columns']
    if scored_columns == 0:
        raise ValueError('Historical NRMSE is undefined: every query column was excluded')
    unused, center, scale = observed_standardize(missing)
    per_column = [float(np.sqrt(np.mean((prediction[selected[:, column], column]
                                        - truth[selected[:, column], column]) ** 2)) / scale[column])
                  for column in range(truth.shape[1]) if selected[:, column].any()]
    return dict(nrmse=result['nrmse_original'], nrmse_observed_sd=float(np.mean(per_column)),
                rmse=result['rmse'], mae=result['mae'], r2=result['r2_columnwise'],
                scored_columns=scored_columns, skipped_constant_columns=result['nrmse_skipped_constant_columns'])


def fixed_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(method, specification):
    from tabimpute.interface import ImputePFN

    path = Path(specification['checkpoint'])
    if method == 'tabimpute':
        from tabimpute.prepreocess import RandomRowColumnPermutation, RandomRowPermutation, RandomColumnPermutation

        preprocessors = [RandomRowColumnPermutation(), RandomRowColumnPermutation(),
                         RandomRowPermutation(), RandomColumnPermutation()]
        return ImputePFN(device='cuda', checkpoint_path=str(path), entry_wise_features=True, nhead=2, use_ema=False,
                         precision_mode='full_bfloat16', prediction_stat='median',
                         max_num_rows=None, max_num_cols=None, max_num_chunks=1, preprocessors=preprocessors, verbose=False)
    config = json.loads(Path(specification['config']).read_text())
    return ImputePFN(device='cuda', checkpoint_path=str(path), entry_wise_features=False,
                     json_config={'config': config['model']}, model_family='new_stable',
                     precision_mode='amp_bfloat16', prediction_stat='mean',
                     max_num_rows=None, max_num_cols=None, max_num_chunks=1, preprocessors=None, verbose=False)


def mice_forest(matrix, seed):
    import miceforest

    frame = pd.DataFrame(matrix, columns=[str(column) for column in range(matrix.shape[1])])
    for parameters in ({}, dict(bagging_fraction=1., bagging_freq=0, feature_fraction=.632)):
        kernel = miceforest.ImputationKernel(frame, num_datasets=5, mean_match_candidates=0, random_state=seed)
        kernel.mice(iterations=5, verbose=False, num_threads=1, **parameters)
        prediction = np.mean([kernel.complete_data(dataset=index).to_numpy() for index in range(5)], axis=0)
        if np.isfinite(prediction).all():
            return prediction
    raise ValueError('Nonfinite MICE-Forest output after the registered no-row-subsampling retry')


def run_r(method, matrix, seed, timeout):
    from grzesiak_benchmark import RSCRIPT, WORK

    with tempfile.TemporaryDirectory(prefix='repeated_r_') as temporary:
        folder = Path(temporary)
        pd.DataFrame(matrix).to_csv(folder / 'input.csv', index=False, na_rep='NA')
        request = dict(upstream=str(WORK / 'upstream'), method=method.removeprefix('r_'), seed=seed,
                       input=str(folder / 'input.csv'), output=str(folder / 'output.csv'), result=str(folder / 'result.json'))
        (folder / 'request.json').write_text(json.dumps(request))
        subprocess.run([str(RSCRIPT), '--vanilla', str(REPO / 'scripts/grzesiak_bridge.R'),
                        str(folder / 'request.json')], check=True, timeout=timeout)
        result = json.loads((folder / 'result.json').read_text())
        if result['status'] != 'ok':
            raise RuntimeError(result['error'])
        return pd.read_csv(folder / 'output.csv').to_numpy(dtype=float)


def impute(method, missing, seed, model=None, timeout=1800):
    fixed_seed(seed)
    normalized, center, scale = observed_standardize(missing)
    empty_columns = np.isnan(normalized).all(axis=0)
    normalized[:, empty_columns] = 0.
    if method in MODEL_PATHS or method == 'tabimpute':
        from evaluate_chunked_uci import configure_windows

        configure_windows(model, normalized.shape, 2000, 1000)
        result = model.impute(normalized.copy())
    elif method == 'column_mean':
        result = np.nan_to_num(normalized, nan=0.)
    elif method == 'knn':
        from sklearn.impute import KNNImputer

        result = KNNImputer(n_neighbors=5, keep_empty_features=True).fit_transform(normalized)
    elif method == 'mice':
        result = mice_forest(normalized, seed)
    elif method in R_METHODS:
        result = run_r(method, normalized, seed, timeout)
    elif method == 'forestdiffusion':
        from ForestDiffusion import ForestDiffusionModel

        result = ForestDiffusionModel(normalized, n_t=50, diffusion_type='vp', n_jobs=1, seed=seed).impute()
    elif method in ('remasker', 'cacti', 'diffputer'):
        import sys

        sys.path.insert(0, str(REPO / 'benchmark'))
        if method == 'remasker':
            from remasker_wrapper import ReMaskerImputer

            estimator = ReMaskerImputer(device='cuda')
        elif method == 'cacti':
            from cacti_wrapper import CACTIImputer

            estimator = CACTIImputer(device='cuda')
        else:
            from diffputer_wrapper import DiffPuterImputer

            estimator = DiffPuterImputer(device='cuda')
        result = estimator.fit_transform(normalized)
    else:
        from hyperimpute.plugins.imputers import Imputers

        normalized[np.isnan(normalized).all(axis=1), :] = 0.
        plugin = 'sinkhorn' if method == 'ot_sinkhorn' else method
        result = Imputers().get(plugin, random_state=seed).fit_transform(pd.DataFrame(normalized)).to_numpy()
    result = np.asarray(result, dtype=np.float64)
    if result.shape != missing.shape or not np.isfinite(result).all():
        raise ValueError(f'{method}: invalid prediction before restoring observed entries')
    result = result * scale + center
    result[np.isfinite(missing)] = missing[np.isfinite(missing)]
    return result
