echo off
cd /d %~dp0
set root=%userprofile%\anaconda3
call %root%\Scripts\activate.bat %root%
call activate kaolin_latent_nerf

call python -m scripts.train_latent_paint --config_path demo_configs/latent_paint/BEST_fish/striped_bass_w_loss.yaml

cmd /k
