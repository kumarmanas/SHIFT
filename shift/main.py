'''
Main run functions.
@author: 
'''
import argparse
from datetime import datetime
import json
import logging
import multiprocessing
import os
from pathlib import Path
import pathlib
import pickle
import sys
import cProfile, pstats, io
from pstats import SortKey
import sys

import gin
from ray import tune
import ray
from ray.tune.registry import register_trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.variant_generator import grid_search
from ray.tune.utils import validate_save_restore
from tqdm import tqdm

from shift.util.config import paramSearch
from shift.util.tune import trial_name_string

#from shift.tf.dataloader.ArgoverseUtils import AV2_LOCATIONS
from shift.tf.dataloader.NuScenesDataLoader import NUSC_LOCATIONS, NUSC_LOCATION_TESTS


def eval_dummy(return_dict, name, args, gin_config):
    """
    Creates an Trainer object for determining the relevant hyperparameters.

    :param dict return_dict: Dictionary that will contain the hyperparameters.
    :param str name: Experiment name, also used for folders
    :param dict args: Program arguments, must contain one of the optimizer flags and dir
    :param str gin_config: Serialized gin config string
    """

    from shift.tf.dataloader.NuScenesDataLoader import NuscenesDataset
    from shift.tf.model.training import Trainer
    from shift.tf.model.models import (
        Covernet,
        CovernetSNGP,
        CovernetHETSNGP
    )
    from shift.util.config import paramSearch

    gin.parse_config(gin_config)
    trainer_dummy = Trainer(dir=args.dir, current_time=name, resume_model=args.resume)
    default_config = {
        k: v
        for k, v in trainer_dummy.hyperparameters.items()
        if not k in ["gin_config", "dir"]
    }
    return_dict.update(default_config)


def run_tune(name, args, target_metric, target_mode, max_t):
    """
    Executes the ray tune optimizer.

    :param str name: Experiment name, also used for folders
    :param dict args: Program arguments, must contain one of the optimizer flags and dir
    :param str target_metric: Metric to optimize
    :param str target_mode: min or max for optimization direction
    :param int max_t: Maximum number of epochs per trial
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    process_eval = multiprocessing.Process(
        target=eval_dummy,
        kwargs={
            "return_dict": return_dict,
            "name": name,
            "args": args,
            "gin_config": gin.config.config_str(),
        },
    )
    process_eval.start()
    process_eval.join()
    default_config = dict(return_dict)

    if args.optimize1:
        space_keys, config, num_trials = paramSearch(
            default_config=default_config, space_name="Sampler"
        )
        scheduler = PopulationBasedTraining(
            perturbation_interval=2, hyperparam_mutations=config
        )
        search_alg = None
        num_samples = 2 if args.test else num_trials
    elif args.optimize2:
        space_keys, search_space, num_trials = paramSearch(
            default_config=default_config, space_name="ConfigSpace"
        )
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=max_t,
            reduction_factor=4,
            stop_last_trials=True,
        )

        search_alg = TuneBOHB(
            metric=target_metric,
            mode=target_mode,
            space=search_space,  # If you want to set the space manually
            max_concurrent=4,
        )
        # Does not work
        # search_alg = TuneBOHBRepeater(search_alg,repeats,set_index=False)

        num_samples = 5 if args.test else num_trials
    elif args.optimize3:
        space_keys, search_space, num_trials = paramSearch(
            default_config=default_config, space_name="Ray"
        )
        search_alg = HEBOSearch(
            space=search_space,
            metric=target_metric,
            mode=target_mode,
            max_concurrent=1 if args.test else 4,
        )
        scheduler = None
        num_samples = 2 if args.test else num_trials

    config = {
            "gin_config": gin.config.config_str(),
            "dir": args.dir,
        }
    if args.init is not None:
        config["load_model"]: args.init
        
    tune_result = tune.run(
        TuneableTrainer,
        name=name,
        resources_per_trial={"cpu": 1} if args.test else {"gpu": 1},
        num_samples=num_samples,
        metric=target_metric,
        checkpoint_score_attr=target_mode + "-" + target_metric,
        checkpoint_freq=1,
        mode=target_mode,
        search_alg=search_alg,
        config=config,
        stop={"training_iteration": max_t},
        verbose=3,
        scheduler=scheduler,
        keep_checkpoints_num=1,
        local_dir=Path(args.dir) / "ray_tune",
        trial_dirname_creator=partial(trial_name_string, consider_keys=space_keys),
    )
    (args.dir / "ray_tune" / name).mkdir(exist_ok=True)
    tune_result.results_df.to_csv(args.dir / "ray_tune" / name / "tune_results.csv")
    with open(args.dir / "ray_tune" / name / "tune_results.pkl", "wb") as file:
        pickle.dump(tune_result, file)
    return tune_result

def rerun_config(best_config, name, args, target_metric, target_mode, max_t):
    """
    Reruns the best found hyperparameter configuration 4 times.

    :param dict best_config: Hyperparameter configuration to use
    :param str name: Experiment name, also used for folders
    :param dict args: Program arguments, must contain dir
    :param str target_metric: Metric to optimize
    :param str target_mode: min or max for optimization direction
    :param int max_t: Maximum number of epochs per run
    """
    tune_result_rerun = tune.run(
        TuneableTrainer,
        name=name,
        resources_per_trial={"cpu": 1} if args.test else {"gpu": 1},
        num_samples=3,
        checkpoint_score_attr=target_mode + "-" + target_metric,
        checkpoint_freq=1,
        config={
            k: v if k in ["gin_config", "dir"] else grid_search([v])
            for k, v in best_config.items()
        },
        stop={"training_iteration": max_t},
        verbose=3,
        keep_checkpoints_num=1,
        local_dir=Path(args.dir) / "ray_tune",
        trial_dirname_creator=partial(trial_name_string, consider_keys=[], is_eval=True)
    )

    tune_result_rerun.results_df.to_csv(
        args.dir / "ray_tune" / name / "rerun_results.csv"
    )
    with open(args.dir / "ray_tune" / name / "rerun_results.pkl", "wb") as file:
        pickle.dump(tune_result_rerun, file)
    return tune_result_rerun


def populate_cache(model_cls="CovernetDet"):
    """
    Fills the cache
    """
    mdl_input_names = globals()[model_cls].mdl_input_names
    mdl_output_names = globals()[model_cls].mdl_output_names
    mp_pool = multiprocessing.Pool(processes=np.minimum(6, multiprocessing.cpu_count()))
    with gin.config_scope('train'):
        train_dataset = NuscenesDataset(mp_pool=mp_pool)
        for _ in tqdm(train_dataset.get_dataset(mdl_input_names,mdl_output_names).batch(1)):
            pass
 
    with gin.config_scope('val'):
        val_dataset = NuscenesDataset(mp_pool=mp_pool)
        val_dataset._helper = train_dataset._helper
        for _ in tqdm(val_dataset.get_dataset(mdl_input_names, mdl_output_names).batch(1)):
            pass
 
    with gin.config_scope('test'):
        test_dataset = NuscenesDataset(mp_pool=mp_pool)
        test_dataset._helper = train_dataset._helper
        for _ in tqdm(test_dataset.get_dataset(mdl_input_names, mdl_output_names).batch(1)):
            pass
 
    mp_pool.terminate()

def single_run_and_eval(args, name, target_metric):
    """
    Performes a single training run and evaluates the best result, if requested (args.eval).

    :param dict args: Program arguments
    :param str name: Experiment name
    :param str target_metric: Metric to minimize (for selecting best result)
    """
    extra_kwargs = {}
    if args.init is not None:
        extra_kwargs["load_model"] = args.init
        
    trainer = Trainer(
        dir=args.dir, current_time=name, resume_model=args.resume, **extra_kwargs
    )
    trainer.build()
    epoch = trainer.start_epoch
    if not args.notrain:
        history = trainer.run_training(lr=trainer.hyperparameters.get("lr", None))
        best_idx = np.nanargmin(history.history[target_metric])
        epoch = epoch + best_idx
        best_mdls = list(trainer.mdl_dir.glob(f'Model_ep{epoch}*.index'))
        if len(best_mdls) != 1:
            print(f'ERROR: Model_ep{epoch}*.index returned {len(best_mdls)} models')
            epoch = epoch + len(history.history[target_metric])
        else:
            trainer = Trainer(
                dir=args.dir,
                current_time=name,
                resume_model=str(best_mdls[0].with_suffix('')),
            )
            trainer.build()
    if args.fisher:
        ckp_path = trainer.mdl_dir / f"Model_ep{epoch}_Fisher_{trainer.fisher_type}"
        trainer.fisher_estimation(ckp_path=ckp_path)
    if args.eval:
        trainer.eval(epoch)


if __name__ == '__main__':
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="gin config filename", required=False, default=None
    )
    parser.add_argument(
        "-j",
        "--json",
        help="json config filename, override gin file",
        required=False,
        default=None,
    )
    parser.add_argument("-b", "--backend", help="torch or tf", default="tf")
    parser.add_argument(
        "-e",
        "--eval",
        help="evalates the result on the test set",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-o1",
        "--optimize1",
        help="optimizer hyperparameters with PBO",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-o2",
        "--optimize2",
        help="optimizer hyperparameters with BOHB",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-o3",
        "--optimize3",
        help="optimizer hyperparameters with HEBO",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-i",
        "--init",
        help="load optimizer results [in o1/o2/o3 mode, as tune_results.pkl] or a rerun_results.pkl or model file, as initial weights (excluding optimizer state).",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Model file for resuming training (including optimizer state). Can not be used with init and not with optimization mode.",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--test",
        help="local test only, reduces runtime",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-d",
        "--dir",
        help="dir for results",
        default=str(Path(os.path.dirname(os.path.realpath(__file__)))),
    )
    parser.add_argument(
        "-p",
        "--populate",
        help="popluate cache",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-n",
        "--notrain",
        help="evaluation only",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-k",
        "--repetitions",
        help="repetitions of run",
        default=1,
        type=int
    )
    parser.add_argument(
        "--resume-tune",
        help="Resume a crashed tune run by specifying the output directory",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--fisher",
        help="estimate fisher relevance",
        default=False,
        action='store_true',
    )
    args = parser.parse_args()

    if args.json is not None:
        with open(args.json, "r") as jsonfile:
            additional_config = json.load(jsonfile)
            for k, v in additional_config.items():
                if isinstance(v, str) and "$" in v:
                    print(f"Expanded {k}: {v}")
                    additional_config[k] = os.path.expandvars(v)
    else:
        additional_config = {}

    assert args.config is not None or (
        "args" in additional_config and "config" in additional_config["args"]
    )

    if "args" in additional_config:
        for k, v in additional_config["args"].items():
            if isinstance(v, str) and "$" in v:
                v = os.path.expandvars(v)
                print(f"Expanded {k}: {v}")
            setattr(args, k, v)

    if args.backend == "torch":
        from shift.pytorch.dataloader.NuScenesDataLoader import NuscenesDataset
        from shift.pytorch.model.training import Trainer
        from shift.pytorch.model.models import *
    if args.backend == "tf":
        from shift.tf.dataloader.NuScenesDataLoader import NuscenesDataset
        #from shift.tf.dataloader.ArgoverseDataLoader import ArgoverseDataLoader
        from shift.tf.model.training import Trainer, TuneableTrainer
        from shift.tf.model.models import *

    register_trainable("TuneableTrainer", TuneableTrainer)
    if 'linux' in sys.platform:
        pathlib.WindowsPath = pathlib.PosixPath
    else:
        pathlib.PosixPath = pathlib.WindowsPath

    gin.parse_config_file(args.config)

    for k, v in additional_config.items():
        if k == "args":
            continue
        gin.bind_parameter(k, v)

    eps_set = gin.config._CONFIG[('', 'shift.tf.model.training.Trainer')].get(
        'eps_set', 4
    )
    gin.config.bind_parameter('NuscenesDataset.eps_set', eps_set)

    multi_label = gin.config._CONFIG[
        ('', 'shift.tf.model.training.Trainer')
    ].get('multi_label', False)
    multi_label = gin.config._CONFIG[
        ('', 'shift.tf.model.training.Trainer')
    ].get('multi_label', False)
    prior_model = gin.config._CONFIG[
        ('', 'shift.tf.model.training.Trainer')
    ].get('prior_model', False)
    multitask_lambda = gin.config._CONFIG[
        ('', 'shift.tf.model.training.Trainer')
    ].get('multitask_lambda', 0)
    transfer_model = gin.config._CONFIG[
        ('', 'shift.tf.model.training.Trainer')
    ].get('load_model', None)
                
    if multitask_lambda > 0 and prior_model is False:
        exp_type = "multitask"
    elif multi_label and prior_model is False:
        exp_type = "knowledge"
    elif not multi_label and prior_model is not False:
        exp_type = "continual"
    elif not multi_label and prior_model is False and transfer_model is None:
        exp_type = "trajectory"
    elif not multi_label and not transfer_model is None:
        exp_type = "transfer"
    else:
        raise NotImplementedError("Invalid combination of multi_label and prior model!")

    model_cls = str(
        gin.config._CONFIG[('', 'shift.tf.model.training.Trainer')].get(
            'model_factory'
        )
    ).replace("@", "")

    dataloader = str(
        gin.config._CONFIG[('', 'shift.tf.model.training.Trainer')].get(
            'dataloader'
        )
    ).replace("@", "")

    args.dir = Path(args.dir) / exp_type / f'Epsilon{int(eps_set)}' / model_cls

    for i in range(args.repetitions):
        if args.resume and args.notrain:
            name = Path(args.resume).parent.parent.name
        else:
            name = datetime.now().strftime('%b%d_%H-%M-%S')

        # if multitask_lambda > 0.0: # multitask and (multilabel or not multilabel)
        #     target_metric = "val_cls_nll"
        # elif multi_label: # multilabel and not multitask
        #     if model_cls == "MultiPathDet" or model_cls == "MultiPathVI":
        #         target_metric = "val_cls_nll"
        #     else:
        #         target_metric = "val_nll"
        # else: # not multilabel and not multitask
        #     target_metric = "val_cls_nll"
        if multi_label:
            target_metric = "val_label_nll"
        else:
            target_metric = "val_cls_nll"
        target_mode = "min"

        if args.populate:
            populate_cache(model_cls)

        if args.optimize1 or args.optimize2 or args.optimize3:
            ray.init(local_mode=not sys.gettrace() is None)

            max_t = gin.config._CONFIG[
                ('', 'shift.tf.model.training.Trainer')
            ].get('epochs', 10)

            ray_logger = logging.getLogger('ray')
            # ray_logger.setLevel(logging.DEBUG)
            (Path(args.dir) / "ray_tune" / name).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(args.dir) / "ray_tune" / name / "ray_output.log")
            ray_logger.addHandler(fh)

            if args.init:
                if isinstance(args.init, str) and args.init.endswith(".pkl"):
                    with open(Path(args.init), "rb") as file:
                        tune_result = pickle.load(file)
                else:
                    tune_result = run_tune(name, args, target_metric, target_mode, max_t)
            else:
                if args.resume_tune:
                    resume_dir = Path(args.resume_tune)
                    if resume_dir.exists():
                        tune_result = run_tune(name, args, target_metric, target_mode, max_t, resume=True)
                        tune_result.local_dir = str(resume_dir)
                    #raise NotImplementedError
                    else:
                        print(f"Resume directory {resume_dir} does not exist. Starting a new experiment.")
                        tune_result = run_tune(name, args, target_metric, target_mode, max_t)
                else:
                    tune_result = run_tune(name, args, target_metric, target_mode, max_t)

            if args.eval:
                best_config = tune_result.get_best_config(
                    metric=target_metric,
                    mode=target_mode,
                    scope="last" if model_cls == "CovernetVI" or model_cls == "MultiPathVI" else "all",
                )
                tune_result_rerun = rerun_config(
                    best_config, name, args, target_metric, target_mode, max_t
                )
                if model_cls == "CovernetVI" or model_cls == "MultiPathVI":
                    args.init = [
                        tune_result_rerun.get_last_checkpoint(
                            trial, metric=target_metric, mode=target_mode
                        )
                        for trial in tune_result_rerun.trials
                    ]
                else:
                    args.init = [
                        tune_result_rerun.get_best_checkpoint(
                            trial, metric=target_metric, mode=target_mode
                        )
                        for trial in tune_result_rerun.trials
                    ]
        else:
            best_config = {}

        trainer_logger = logging.getLogger('trainer')
        trainer_logger.setLevel(logging.DEBUG)
        (Path(args.dir) / 'logs' / name).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(args.dir) / 'logs' / name / "trainer_output.log")
        trainer_logger.addHandler(fh)

        if args.init:

            if isinstance(args.init, str) and '*' in args.init:
                root = Path(args.init)
                checkpoints = sorted(
                    list(root.parent.glob(root.name + "/*/checkpoint")), reverse=True
                )
                args.init = []
                found = []
                for path in checkpoints:
                    if not path.parent.parent in found:
                        found.append(path.parent.parent)
                        args.init.append(str(path))

            if not args.notrain:
                if isinstance(args.init, str) and args.init.endswith(".pkl"):
                    with open(Path(args.init), "rb") as file:
                        tune_result_rerun = pickle.load(file)
                    args.init = [
                        tune_result_rerun.get_best_checkpoint(
                            trial, metric=target_metric, mode=target_mode
                        )
                        for trial in tune_result_rerun.trials
                    ]

            best_config['dir'] = Path(args.dir)
            if 'gin_config' in best_config:
                del best_config['gin_config']
            if 'load_model' in best_config:
                del best_config['load_model']
            if isinstance(args.init, list):
                for i, mdl in enumerate(args.init):
                    trainer = Trainer(
                        load_model=mdl, current_time=name, name=f"model_{i}", **best_config
                    )
                    trainer.build()
                    if args.eval:
                        trainer.eval(trainer.start_epoch)
            else:
                single_run_and_eval(args, name, target_metric)

        elif not (args.optimize1 or args.optimize2 or args.optimize3):
            single_run_and_eval(args, name, target_metric)
