echo off
cd /d %~dp0
set root=%userprofile%\anaconda3
call %root%\Scripts\activate.bat %root%
call activate kaolin_latent_nerf

call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/bowfin_fish.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/croaker.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/goldfish.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/koi.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/queen_angelfish.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/regal_blue_tang.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/striped_bass.yaml
call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/our_model/tuna.yaml

cmd /k
