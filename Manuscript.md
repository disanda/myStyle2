# Manuscript

## Train

python run_training.py my_settings.yaml --gpu 0 --resume
> --resume will look in the checkpoint directory if we specified one and attempt to load the latest checkpoint before continuing to train. 

## Generating Images

python run_convert_from_tf.py --download ffhq-config-f --output G.pth D.pth Gs.pth
> Train a network or convert a pretrained one. Example of converting pretrained ffhq model:
--download：
car-config-e
car-config-f
cat-config-f
church-config-f
ffhq-config-e
ffhq-config-f
horse-config-f
car-config-e-Gorig-Dorig
car-config-e-Gorig-Dresnet
car-config-e-Gorig-Dskip
car-config-e-Gresnet-Dorig
car-config-e-Gresnet-Dresnet
car-config-e-Gresnet-Dskip
car-config-e-Gskip-Dorig
car-config-e-Gskip-Dresnet
car-config-e-Gskip-Dskip
ffhq-config-e-Gorig-Dorig
ffhq-config-e-Gorig-Dresnet
ffhq-config-e-Gorig-Dskip
ffhq-config-e-Gresnet-Dorig
ffhq-config-e-Gresnet-Dresnet
ffhq-config-e-Gresnet-Dskip
ffhq-config-e-Gskip-Dorig
ffhq-config-e-Gskip-Dresnet
ffhq-config-e-Gskip-Dskip

python run_generator.py generate_images --network=Gs.pth --seeds=6600-6625 --truncation_psi=0.5
> Generate ffhq uncurated images (matches paper Figure 12)

python run_generator.py generate_images --network=Gs.pth --seeds=66,230,389,1518 --truncation_psi=1.0
> Generate ffhq curated images (matches paper Figure 11)

python run_convert_from_tf.py --download car-config-f --output G_car.pth D_car.pth Gs_car.pth
> Example of converting pretrained car model:

python run_generator.py generate_images --network=Gs_car.pth --seeds=6000-6025 --truncation_psi=0.5
> Generate uncurated car images (matches paper Figure 12)

python run_generator.py style_mixing_example --network=Gs.pth --row_seeds=85,100,75,458,1500 --col_seeds=55,821,1789,293 --truncation_psi=1.0
> Generate style mixing example (matches style mixing video clip)



## Project 

python run_projector.py project_generated_images --network=Gs.pth --seeds=0,1,5
>generated images

python run_projector.py project_real_images --network=Gs.pth --data-dir=path/to/image_folder
>real images