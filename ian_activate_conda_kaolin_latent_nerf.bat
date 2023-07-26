echo off
cd /d %~dp0
set root=%userprofile%\anaconda3
call %root%\Scripts\activate.bat %root%
call activate kaolin_latent_nerf

cmd /k
