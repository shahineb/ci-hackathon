"""
Runs training of experiment

Usage: run_training.py --cfg=<config_file_path>  --o=<output_dir> [--device=<execution_device>] [--experiment_name=<name>] [--seed=<random_seed>]

Options:
  --cfg=<config_file_path>  Path to experiment configuration file
  --o=<output_directory>    Path to experiments output directory
  --device=<gpus_ids>       Ids of GPUs to run training on, if None runs on CPU
  --experiment_name=<name>  Custom naming for subdirectory where experiment outputs are saved
  --seed=<random_seed>      Random seed to use for experiment initialization
"""
import os
from docopt import docopt
import pytorch_lightning as pl
from src.experiments import cGANCloudTOPtoRGB, Logger
from src.models import Unet, PatchGAN
from src.datasets import CloudTOPtoRGBDataset
from src.utils import load_yaml


def main(args, cfg):
    # Set seed for reproducibility
    set_seed(args, cfg)

    # Build experiment
    experiment = build_experiment(cfg)

    # Build logging and callbacks
    logger = make_logger(args)
    model_checkpoint = make_model_checkpoint(cfg['model_checkpoint'])

    # Instantiate trainer instance
    params = {'logger': logger,
              'checkpoint_callback': model_checkpoint,
              'precision': cfg['experiment']['precision'],
              'max_epochs': cfg['experiment']['max_epochs'],
              'gpus': args['--device']}
    trainer = pl.Trainer(**params)

    # Run training
    trainer.fit(experiment)


def build_experiment(cfg):
    generator = Unet.build(cfg['generator'])
    discriminator = PatchGAN.build(cfg['discriminator'])
    dataset = CloudTOPtoRGBDataset.build(cfg['dataset'])

    experiment = cGANCloudTOPtoRGB(generator=generator,
                                   discriminator=discriminator,
                                   dataset=dataset,
                                   split=list(cfg['dataset']['split'].values()),
                                   optimizer_kwargs=cfg['optimizer'],
                                   lr_scheduler_kwargs=cfg['lr_scheduler'],
                                   dataloader_kwargs=cfg['dataset']['dataloader'],
                                   seed=cfg['experiment']['seed'])
    return experiment


def set_seed(args, cfg):
    """Sets random seed for reproducibility
    """
    seed = int(args["--seed"]) if args["--seed"] else cfg['experiment']['seed']
    pl.seed_everything(seed)


def make_logger(args):
    """Build logger instance pointing to specified output directory
    """
    logger_kwargs = {'save_dir': os.path.dirname(args['--o']),
                     'name': os.path.basename(args['--o'])}

    if args['--experiment_name']:
        logger_kwargs.update({'version': args['--experiment_name']})

    logger = Logger(**logger_kwargs)
    return logger


def make_model_checkpoint(cfg):
    """Build model checkpointing callback
    """
    model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg)
    return model_checkpoint


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load configuration file
    cfg = load_yaml(args["--cfg"])

    # Run training
    main(args, cfg)
