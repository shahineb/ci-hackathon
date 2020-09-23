"""
Runs inference on testing set

Usage: run_inference.py --cfg=<config_file_path> --chkpt=<path_to_checkpoint>  --o=<output_path>

Options:
  --cfg=<config_file_path>      Path to experiment configuration file
  --chkpt=<path_to_checkpoint>  Path to model checkpoint to load
  --o=<output_path>             Path to inference output file
"""
from docopt import docopt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from src.experiments import cGANCloudTOPtoRGB
from src.models import Unet, PatchGAN
from src.datasets import CloudTOPtoRGBDataset
from src.utils import load_yaml


def main(args, cfg):
    experiment = build_experiment(args, cfg)
    to_tensor = ToTensor()
    Xtest = np.load('data/X_test_CI20_phase1.npy')

    predictions = np.zeros_like(Xtest)
    for i, x in enumerate(Xtest):
        with torch.no_grad():
            prediction = experiment.model(to_tensor(x).unsqueeze(0).cuda())
        predictions[i] = prediction.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    np.save(args['--o'], predictions)


def build_experiment(args, cfg):
    generator = Unet.build(cfg['model']['generator'])
    discriminator = PatchGAN.build(cfg['model']['discriminator'])
    dataset = CloudTOPtoRGBDataset.build(cfg['dataset'])

    experiment = cGANCloudTOPtoRGB.load_from_checkpoint(generator=generator.eval(),
                                                        discriminator=discriminator.eval(),
                                                        dataset=dataset,
                                                        split=list(cfg['dataset']['split'].values()),
                                                        optimizer_kwargs=cfg['optimizer'],
                                                        lr_scheduler_kwargs=cfg['lr_scheduler'],
                                                        dataloader_kwargs=cfg['dataset']['dataloader'],
                                                        supervision_weight=cfg['experiment']['supervision_weight'],
                                                        seed=cfg['experiment']['seed'],
                                                        checkpoint_path=args['--chkpt'])
    return experiment


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load configuration file
    cfg = load_yaml(args["--cfg"])

    # Run training
    main(args, cfg)
