import stylegan2
import torch
from stylegan2 import utils
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#G = stylegan2.models.load('D:\\AI-Lab\\StyleGANs\\stylegan2_pytorch_Tetratrio\\pre-model\\ffhq-config-f\\Gs.pth').to(device)
G = stylegan2.models.load('D:\\AI-Lab\\StyleGANs\\stylegan2_pytorch_Tetratrio\\pre-model-trans\\ffhq-config-e-Gorig-Dskip\\Gs.pth').to(device)
G.eval()

latents,noise = torch.load('./wwm_w_wz_0.pt')
G.static_noise(noise_tensors=noise)
latents = torch.tensor(latents).to(device)
latents = latents.unsqueeze(0)
G_s = G.G_synthesis

#----------origin---------
#generated = G_s(latents)
#images = utils.tensor_to_PIL(generated, pixel_min=-1, pixel_max=1)
#images[0].save('./wwm_noise.png')


direction = np.load('./direction/age.npy')
direction = torch.tensor(direction).to(device)
for i, coeff in enumerate([-5, -1.5, 0, 1.5, 5]):
        new_latent = latents
        new_latent[:8] = (latents + coeff*direction)[:8]
        generated = G_s(new_latent)
        images = utils.tensor_to_PIL(generated, pixel_min=-1, pixel_max=1)
        images[0].save('./trump_%d.png'%i)

