import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.experiments.experiment import ImageTranslationExperiment
from src.losses import SSIM
from .utils import process_tensor_for_vis


class cGANCloudTOPtoRGB(ImageTranslationExperiment):
    """Setup to train  conditional GANs at generating optical RGB images from
        IR cloud top inputs

                             +-----------+
            Cloud TOP -----> | Generator |---> Predicted_RGB_Image
                             +-----------+


    Adversarial networks loss computation given by :
        LossDisc = E_{x~realdata}[-logD(x)] + E_{z~inputs}[-log(1 - D(G(z)))]
        LossGen = E_{z~inputs}[-logD(z)]

    We approximate:
        E_{x~realdata}[-logD(x)] = Avg(CrossEnt_{x:realbatch}(1, D(x)))
        E_{z~inputs}[-log(1 - D(G(z)))] = Avg(CrossEnt_{x:fakebatch}(0, D(x)))
        E_{z~inputs}[-logD(z)] = Avg(CrossEnt_{x:fakebatch}(1, D(x)))


    Args:
        generator (nn.Module)
        discriminator (nn.Module)
        dataset (MODISLandsatReflectanceFusionDataset)
        split (list[float]): dataset split ratios in [0, 1] as [train, val]
            or [train, val, test]
        supervision_weight_l1 (float): weight supervision loss term for l1 loss
        supervision_weight_ssim (float): weight supervision loss term for ssim loss
        dataloader_kwargs (dict): parameters of dataloaders
        optimizer_kwargs (dict): parameters of optimizer defined in LightningModule.configure_optimizers
        lr_scheduler_kwargs (dict): paramters of lr scheduler defined in LightningModule.configure_optimizers
        seed (int): random seed (default: None)
    """
    def __init__(self, generator, discriminator, dataset, split, dataloader_kwargs,
                 optimizer_kwargs, lr_scheduler_kwargs=None, supervision_weight_l1=None,
                 supervision_weight_ssim=None, seed=None):
        super().__init__(model=generator,
                         dataset=dataset,
                         split=split,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler_kwargs=lr_scheduler_kwargs,
                         criterion=nn.BCELoss(),
                         seed=seed)
        self.supervision_weight_l1 = supervision_weight_l1
        self.supervision_weight_ssim = supervision_weight_ssim
        self.discriminator = discriminator
        self.ssim_criterion = SSIM()

    def forward(self, x):
        return self.generator(x)

    def train_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Instantiate loader
        train_loader_kwargs = self.dataloader_kwargs.copy()
        train_loader_kwargs.update({'dataset': self.train_set,
                                    'shuffle': True})
        loader = DataLoader(**train_loader_kwargs)
        return loader

    def val_dataloader(self):
        """Implements LightningModule train loader building method
        """
        # Instantiate loader
        val_loader_kwargs = self.dataloader_kwargs.copy()
        val_loader_kwargs.update({'dataset': self.val_set})
        loader = DataLoader(**val_loader_kwargs)
        return loader

    def configure_optimizers(self):
        """Implements LightningModule optimizer and learning rate scheduler
        building method
        """
        # Separate optimizers for generator and discriminator
        gen_optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs['generator'])
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), **self.optimizer_kwargs['discriminator'])

        # Separate learning rate schedulers
        gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                                  **self.lr_scheduler_kwargs['generator'])
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer,
                                                                   **self.lr_scheduler_kwargs['discriminator'])

        # Make lightning output dictionnary fashion
        gen_optimizer_dict = {'optimizer': gen_optimizer, 'scheduler': gen_lr_scheduler, 'frequency': 1}
        disc_optimizer_dict = {'optimizer': disc_optimizer, 'scheduler': disc_lr_scheduler, 'frequency': 2}
        return gen_optimizer_dict, disc_optimizer_dict

    def _step_generator(self, source, target):
        """Runs generator forward pass and loss computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on source domain data
        pred_target = self(source)
        output_fake_sample = self.discriminator(pred_target, source)

        # Compute generator fooling power
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)

        # Compute image quality metrics
        psnr, ssim, sam = self._compute_iqa_metrics(pred_target, target)

        # Compute L1 regularization term
        mae = F.smooth_l1_loss(pred_target, target)
        ssim_loss = 1 - self.ssim_criterion(pred_target, target)
        return gen_loss, ssim_loss, mae, psnr, ssim, sam

    def _step_discriminator(self, source, target):
        """Runs discriminator forward pass, loss computation and classification
        metrics computation

        Args:
            source (torch.Tensor): (batch_size, C, H, W) tensor
            target (torch.Tensor): (batch_size, C, H, W) tensor

        Returns:
            type: dict
        """
        # Forward pass on target domain data
        output_real_sample = self.discriminator(target, source)

        # Compute discriminative power on real samples - label smoothing on positive samples
        target_real_sample = 0.8 + 0.2 * torch.rand_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)

        # Generate fake sample + forward pass, we detach fake samples to not backprop though generator
        pred_target = self(source)
        output_fake_sample = self.discriminator(pred_target.detach(), source)

        # Compute discriminative power on fake samples
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample

        # Compute classification training metrics
        fooling_rate, precision, recall = self._compute_classification_metrics(output_real_sample, output_fake_sample)
        return disc_loss, fooling_rate, precision, recall

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Implements LightningModule training logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)
            optimizer_idx (int): {0: gen_optimizer, 1: disc_optimizer}

        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch

        # Run either generator or discriminator training step
        if optimizer_idx == 0:
            gen_loss, ssim_loss, mae, psnr, ssim, sam = self._step_generator(source, target)
            logs = {'Loss/train_generator': gen_loss,
                    'Loss/train_ssim_loss': ssim_loss,
                    'Loss/train_mae': mae,
                    'Metric/train_psnr': psnr,
                    'Metric/train_ssim': ssim,
                    'Metric/train_sam': sam}
            loss = gen_loss + self.supervision_weight_l1 * mae + self.supervision_weight_ssim * ssim_loss

        if optimizer_idx == 1:
            disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)
            logs = {'Loss/train_discriminator': disc_loss,
                    'Metric/train_fooling_rate': fooling_rate,
                    'Metric/train_precision': precision,
                    'Metric/train_recall': recall}
            loss = disc_loss

        # Make lightning fashion output dictionnary
        output = {'loss': loss,
                  'progress_bar': logs,
                  'log': logs}
        return output

    def on_epoch_end(self):
        """Implements LightningModule end of epoch operations
        """
        # Compute generated samples out of logging images
        source, target = self.logger._logging_images
        with torch.no_grad():
            output = self(source)

        if self.current_epoch == 0:
            # Log input and groundtruth once only at first epoch
            self.logger.log_images(process_tensor_for_vis(source, 1, 99), tag='Source - Cloud TOP', step=self.current_epoch)
            self.logger.log_images(process_tensor_for_vis(target, 1, 99), tag='Target - Visible RGB', step=self.current_epoch)

        # Log generated image at current epoch
        self.logger.log_images(process_tensor_for_vis(output, 1, 99), tag='Generated - Visible RGB', step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """Implements LightningModule validation logic

        Args:
            batch (tuple[torch.Tensor]): source, target pairs batch
            batch_idx (int)

        Returns:
            type: dict
        """
        # Unfold batch
        source, target = batch

        # Store into logger images for visualization
        if not hasattr(self.logger, '_logging_images'):
            self.logger._logging_images = source, target

        # Run forward pass on generator and discriminator
        gen_loss, ssim_loss, mae, psnr, ssim, sam = self._step_generator(source, target)
        disc_loss, fooling_rate, precision, recall = self._step_discriminator(source, target)

        # Encapsulate scores in torch tensor
        output = torch.Tensor([gen_loss, ssim_loss, mae, psnr, ssim, sam, disc_loss, fooling_rate, precision, recall])
        return output

    def validation_epoch_end(self, outputs):
        """LightningModule validation epoch end hook

        Args:
            outputs (list[torch.Tensor]): list of validation steps outputs

        Returns:
            type: dict
        """
        # Average loss and metrics
        outputs = torch.stack(outputs).mean(dim=0)
        gen_loss, ssim_loss, mae, psnr, ssim, sam, disc_loss, fooling_rate, precision, recall = outputs

        # Make tensorboard logs and return
        logs = {'Loss/val_generator': gen_loss.item(),
                'Loss/val_discriminator': disc_loss.item(),
                'Loss/val_ssim_loss': ssim_loss.item(),
                'Loss/val_mae': mae.item(),
                'Metric/val_psnr': psnr.item(),
                'Metric/val_ssim': ssim.item(),
                'Metric/val_sam': sam.item(),
                'Metric/val_fooling_rate': fooling_rate.item(),
                'Metric/val_precision': precision.item(),
                'Metric/val_recall': recall.item()}

        # Make lightning fashion output dictionnary - track discriminator max loss for validation
        output = {'val_loss': ssim_loss,
                  'log': logs,
                  'progress_bar': logs}
        return output

    @property
    def generator(self):
        return self.model

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def supervision_weight_l1(self):
        return self._supervision_weight_l1

    @property
    def supervision_weight_ssim(self):
        return self._supervision_weight_ssim

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    @supervision_weight_l1.setter
    def supervision_weight_l1(self, supervision_weight_l1):
        self._supervision_weight_l1 = supervision_weight_l1

    @supervision_weight_ssim.setter
    def supervision_weight_ssim(self, supervision_weight_ssim):
        self._supervision_weight_ssim = supervision_weight_ssim
