"""
Runs inference on testing set

Usage: run_inference.py --cfg=<config_file_path> --data=<path_to_input_data> --chkpt=<path_to_checkpoint>  --o=<output_path>

Options:
  --cfg=<config_file_path>      Path to experiment configuration file
  --data=<path_to_input_data>   Path to numpy input data file to run inference on
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
    X = np.load(args['--data'])

    predictions = np.zeros_like(X)
    for i, x in enumerate(X):
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
                                                        supervision_weight_l1=cfg['experiment']['supervision_weight_l1'],
                                                        supervision_weight_ssim=cfg['experiment']['supervision_weight_ssim'],
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
