import stylegan2
import torch
from stylegan2 import utils
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#G = stylegan2.models.load('D:\\AI-Lab\\StyleGANs\\stylegan2_pytorch_Tetratrio\\pre-model\\ffhq-config-f\\Gs.pth').to(device)
# G = stylegan2.models.load('D:\\AI-Lab\\StyleGANs\\stylegan2_pytorch_Tetratrio\\pre-model-trans\\ffhq-config-e-Gorig-Dskip\\Gs.pth').to(device)
# G.eval()

# latents,noise = torch.load('./wwm_w_wz_0.pt')
# G.static_noise(noise_tensors=noise)
# latents = torch.tensor(latents).to(device)
# latents = latents.unsqueeze(0)
# G_s = G.G_synthesis

#----------origin latens2img---------
#generated = G_s(latents)
#images = utils.tensor_to_PIL(generated, pixel_min=-1, pixel_max=1)
#images[0].save('./wwm_noise.png')


# direction = np.load('./direction/age.npy')
# direction = torch.tensor(direction).to(device)
# for i, coeff in enumerate([-5, -1.5, 0, 1.5, 5]):
#         new_latent = latents
#         new_latent[:8] = (latents + coeff*direction)[:8]
#         generated = G_s(new_latent)
#         images = utils.tensor_to_PIL(generated, pixel_min=-1, pixel_max=1)
#         images[0].save('./trump_%d.png'%i)

#-----------model initiation------------
common_kwargs = dict(
        data_channels=3,
        base_shape=(4,4),
        conv_filter=[1, 3, 3, 1],
        skip_filter=[1, 3, 3, 1],
        kernel_size=3,
        conv_pad_mode='constant',
        conv_pad_constant=0,
        filter_pad_mode='constant',
        filter_pad_constant=0,
        pad_once=True,
        weight_scale=True
    )

# G_M = stylegan2.models.GeneratorMapping(
#             latent_size=512,
#             label_size=0,
#             num_layers=8,
#             hidden=512,
#             activation='leaky=0.2',
#             normalize_input=True,
#             lr_mul=0.01,
#             weight_scale=True)

# G_S = stylegan2.models.GeneratorSynthesis(
#             channels=[32, 64, 128, 256, 512, 512, 512, 512],
#             latent_size=512,
#             demodulate=True, # g_normalize
#             modulate_data_out=True,
#             conv_block_size=2,
#             activation='leaky=0.2',
#             conv_resample_mode='FIR',
#             skip_resample_mode='FIR',
#             resnet=False,
#             skip=True,
#             fused_resample=True,#g_fused_conv
#             **common_kwargs
#         )

D = stylegan2.models.Discriminator(
            channels= [32, 64, 128, 256, 512, 512, 512, 512],
            label_size= 0,
            conv_block_size= 2,
            activation= 'leaky=0.2',
            conv_resample_mode='FIR',
            skip_resample_mode='FIR',
            mbstd_group_size=0, # 0 is close
            resnet=True,
            skip=False,
            fused_resample=True,
            **common_kwargs
        )

x = torch.randn(4,3,1024,1024)
with torch.no_grad():
	z = D(x)
print(z.shape)


