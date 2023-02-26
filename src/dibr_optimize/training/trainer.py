import sys
from pathlib import Path
from typing import Any, Dict, Union, List

import imageio
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src import utils
from src.dibr_optimize.configs.train_config import TrainConfig
from src.dibr_optimize.training.views_dataset import ViewsDataset
from src.stable_diffusion import StableDiffusion
from src.utils import make_path, tensor2numpy
from src.dibr_optimize.models.textured_mesh import TexturedMeshModel

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.mesh_model = self.init_mesh_model()
        ##self.diffusion = self.init_diffusion()
        self.gt_image = self.init_gt_image()
        ##self.text_z = self.calc_text_embeddings()
        ##self.text_z_side = self.calc_side_only_text_embeddings()
        
        self.optimizer = self.init_optimizer()
        self.dataloaders = self.init_dataloaders()

        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> TexturedMeshModel:
        ##if self.cfg.render.backbone == 'texture-rgb-mesh':
        model = TexturedMeshModel(self.cfg, device=self.device, render_grid_size=self.cfg.render.train_grid_size,
                                    latent_mode=False, texture_resolution=self.cfg.guide.texture_resolution).to(self.device)

        model = model.to(self.device)
        logger.info(
            f'Loaded RGB Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_gt_image(self) -> torch.Tensor:
        import torch
        import torchvision
        from torchvision.io import read_image
        import torchvision.transforms as T
        loaded_image = read_image(self.cfg.guide.gt_image_path)
        print(f"gt_image.size = {loaded_image.size()}")
        return loaded_image

    # def init_diffusion(self) -> StableDiffusion:
    #     diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
    #                                       concept_name=self.cfg.guide.concept_name,
    #                                       latent_mode=self.mesh_model.latent_mode)
    #     for p in diffusion_model.parameters():
    #         p.requires_grad = False
    #     return diffusion_model
    
    # # ian: calculate
    # def calc_side_only_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
    #     ref_text = self.cfg.guide.text
    #     text = f"{ref_text}, (side view)"
    #     text_z_side = self.diffusion.get_text_embeds([text])
    #     return text_z_side

    def init_optimizer(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
        return optimizer

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = ViewsDataset(self.cfg.render, device=self.device, type='train', size=100).dataloader()
        val_loader = ViewsDataset(self.cfg.render, device=self.device, type='val',
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device, type='val',
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                self.train_step += 1
                pbar.update(1)
                # ian: reset gradient
                self.optimizer.zero_grad()
                # ian: render image and send to diffusion to calculate the loss
                # ian: is "pred_rgbs" rendered image or diffused image?
                pred_rgbs, loss = self.train_render(data)
                # ian: optimize according to the render
                # ian: how to calc the gradient? where are the parameters??
                # ian: A: the parameters are defined in the constructor!
                self.optimizer.step()

                # ian: just save some stuff
                if self.train_step % self.cfg.log.save_interval == 0:
                    ## self.save_checkpoint(full=True)
                    # ian: render with multiple views with current texture. frames and texture are saved.
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    # ian: set model to train mode
                    self.mesh_model.train()
                    # ian: save the rendered image
                    self.log_train_renders(pred_rgbs)

                
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        # ian: export mesh
        self.full_eval()
        logger.info('\tDone!')

    # ian: render frames with different perspectives imported from dataloader
    # ian: save rendered scenes and a texture of the mesh
    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        # ian: set model to eval mode
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        for i, data in enumerate(dataloader):
            # ian: render a frame according to the camera settings(data)
            # ian: preds = rendered frame, textures = the texture of the mesh
            preds, textures = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            Image.fromarray(pred).save(save_path / f"step_{self.train_step:05d}_{i:04d}_rgb.png")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.train_step:05d}_texture.png")

        logger.info('Done!')

    def full_eval(self):
        try:
            #self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)
            logger.info("ian: val_large and save_as_video is skipped because it always fails.")
        except:
            logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def train_render(self, data: Dict[str, Any]):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']

        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        pred_rgb = outputs['image']

        print(f"size of pred_rgb = {pred_rgb.size()}")

        # ian: IMPORTANT
        # Guidance loss
        loss_guidance = self.train_step_loss(pred_rgb)
        loss = loss_guidance

        return pred_rgb, loss

    # ian: implement our loss calculation instead of using diffusion.train_step()
    def train_step_loss(self, inputs, guidance_scale=100):
        
        # interp to 512x512 to be fed into vae.
        latents = inputs
        print(f"\nlatents.size() = {latents.size()}\n", flush=True)

        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        ##w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        w = 1.0
        grad = w * (latents - self.gt_image)
        # (latent version)ian: grad.size() = [1, 4, 64, 64]

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        latents.backward(gradient=grad, retain_graph=True)

        return 0 # dummy loss value

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         test=True ,dims=(dim,dim))
        pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        return pred_rgb, texture_rgb

    def log_train_renders(self, preds: torch.Tensor):
        # if self.mesh_model.latent_mode:
        #     pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
        # else:
        #     pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)   # [1, 3, H, W]
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred_rgb = tensor2numpy(pred_rgb[0])

        Image.fromarray(pred_rgb).save(save_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        def decode_texture_img(latent_texture_img):
            decoded_texture = self.diffusion.decode_latents(latent_texture_img)
            decoded_texture = F.interpolate(decoded_texture,
                                            (self.cfg.guide.texture_resolution, self.cfg.guide.texture_resolution),
                                            mode='bilinear', align_corners=False)
            return decoded_texture

        if 'model' not in checkpoint_dict:
            if not self.mesh_model.latent_mode:
                # initialize the texture rgb image from the latent texture image
                checkpoint_dict['texture_img_rgb_finetune'] = decode_texture_img(checkpoint_dict['texture_img'])
            self.mesh_model.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return

        if not self.mesh_model.latent_mode:
            # initialize the texture rgb image from the latent texture image
            checkpoint_dict['model']['texture_img_rgb_finetune'] = \
            decode_texture_img(checkpoint_dict['model']['texture_img'])

        missing_keys, unexpected_keys = self.mesh_model.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        if model_only:
            return

        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

    # def save_checkpoint(self, full=False):

    #     name = f'step_{self.train_step:06d}'

    #     state = {
    #         'train_step': self.train_step,
    #         'checkpoints': self.past_checkpoints,
    #     }

    #     if full:
    #         state['optimizer'] = self.optimizer.state_dict()

    #     state['model'] = self.mesh_model.state_dict()

    #     file_path = f"{name}.pth"

    #     self.past_checkpoints.append(file_path)

    #     if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
    #         old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
    #         print("Should delete file: ", old_ckpt, " but the unlink function is in python38")
    #         ##old_ckpt.unlink(missing_ok=True)

    #     torch.save(state, self.ckpt_path / file_path)
