echo off
cd /d %~dp0
set root=%userprofile%\anaconda3
call %root%\Scripts\activate.bat %root%
call activate kaolin_latent_nerf

call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/bowfin_fish.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/croaker.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/goldfish.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/koi.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/queen_angelfish.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/regal_blue_tang.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/striped_bass.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/tuna.yaml
call python -m scripts.train_latent_nerf --config_path demo_configs/latent_nerf/goldfish_unconstrained.yaml

cmd /k
