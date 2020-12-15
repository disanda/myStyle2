import warnings
import os
import torch
from torch import multiprocessing as mp
import stylegan2
from stylegan2 import utils
from stylegan2.external_models import inception, lpips
from stylegan2.metrics import fid, ppl
from defaults import get_cfg_defaults

#----------------------------------------------------------------------------

def get_dataset(args):
    assert args.data_dir, '--data_dir has to be specified.'
    height, width = [shape * 2 ** (len(args.d_channels or args.channels) - 1) for shape in args.base_shape]# 4 * 2**(n-1)
    dataset = utils.ImageFolder(
        args.data_dir,
        mirror=args.mirror_augment,
        pixel_min=args.pixel_min,
        pixel_max=args.pixel_max,
        height=height,
        width=width,
        resize=args.data_resize,
        grayscale=args.data_channels == 1
    )
    assert len(dataset), 'No images found at {}'.format(args.data_dir)
    return dataset

#----------------------------------------------------------------------------

def get_models(args):
    common_kwargs = dict(
        data_channels=args.data_channels,
        base_shape=args.base_shape,
        conv_filter=args.filter,
        skip_filter=args.filter,
        kernel_size=args.kernel_size,
        conv_pad_mode=args.pad_mode,
        conv_pad_constant=args.pad_constant,
        filter_pad_mode=args.filter_pad_mode,
        filter_pad_constant=args.filter_pad_constant,
        pad_once=args.pad_once,
        weight_scale=args.weight_scale
    )

    if args.g_file:
        G = stylegan2.models.load(args.g_file)
        assert isinstance(G, stylegan2.models.Generator), \
                '`--g_file` should specify a generator model, found {}'.format(type(G))
    else:

        G_M = stylegan2.models.GeneratorMapping(
            latent_size=args.latent,
            label_size=args.label,
            num_layers=args.latent_mapping_layers,
            hidden=args.latent,
            activation=args.g_activation,
            normalize_input=args.normalize_latent,
            lr_mul=args.latent_mapping_lr_mul,
            weight_scale=args.weight_scale
        )

        G_S = stylegan2.models.GeneratorSynthesis(
            channels=args.g_channels or args.channels,
            latent_size=args.latent,
            demodulate=args.g_normalize,
            modulate_data_out=args.modulate_rgb,
            conv_block_size=args.g_conv_block_size,
            activation=args.g_activation,
            conv_resample_mode=args.g_conv_resample_mode,
            skip_resample_mode=args.g_skip_resample_mode,
            resnet=args.g_resnet,
            skip=args.g_skip,
            fused_resample=args.g_fused_conv,
            **common_kwargs
        )

        G = stylegan2.models.Generator(G_mapping=G_M, G_synthesis=G_S)

    if args.d_file:
        D = stylegan2.models.load(args.d_file)
        assert isinstance(D, stylegan2.models.Discriminator), \
                '`--d_file` should specify a discriminator model, found {}'.format(type(D))
    else:
        D = stylegan2.models.Discriminator(
            channels=args.d_channels or args.channels,
            label_size=args.label,
            conv_block_size=args.d_conv_block_size,
            activation=args.d_activation,
            conv_resample_mode=args.d_conv_resample_mode,
            skip_resample_mode=args.d_skip_resample_mode,
            mbstd_group_size=args.group_size,
            resnet=args.d_resnet,
            skip=args.d_skip,
            fused_resample=args.d_fused_conv,
            **common_kwargs
        )
    assert len(G.G_synthesis.channels) == len(D.channels), \
        'While the number of channels for each layer can ' + \
        'differ between generator and discriminator, the ' + \
        'number of layers have to be the same. Received ' + \
        '{} generator layers and {} discriminator layers.'.format(
            len(G.G_synthesis.channels), len(D.channels))

    return G, D

#----------------------------------------------------------------------------

def get_trainer(args):
    dataset = get_dataset(args)
    if args.resume and stylegan2.train._find_checkpoint(args.checkpoint_dir):
        trainer = stylegan2.train.Trainer.load_checkpoint(
            args.checkpoint_dir,
            dataset,
            device=args.device,
            tensorboard_log_dir=args.tensorboard_log_dir
        )
    else:
        G, D = get_models(args)
        trainer = stylegan2.train.Trainer(
            G=G,
            D=D,
            latent_size=args.latent,
            dataset=dataset,
            device=args.device,
            batch_size=args.batch_size,
            label_size=args.label,
            data_workers=args.data_workers,
            G_loss=args.g_loss,
            D_loss=args.d_loss,
            G_reg=args.g_reg,
            G_reg_interval=args.g_reg_interval,
            G_opt_kwargs={'lr': args.g_lr, 'betas': args.g_betas},
            D_reg=args.d_reg,
            D_reg_interval=args.d_reg_interval,
            D_opt_kwargs={'lr': args.d_lr, 'betas': args.d_betas},
            style_mix_prob=args.style_mix,
            G_iter=args.g_iter,
            D_iter=args.d_iter,
            tensorboard_log_dir=args.tensorboard_log_dir,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            half=args.half,
        )
    if args.fid_interval:
        fid_model = inception.InceptionV3FeatureExtractor(
            pixel_min=args.pixel_min, pixel_max=args.pixel_max)
        trainer.register_metric(
            name='FID (299x299)',
            eval_fn=fid.FID(
                trainer.Gs,
                trainer.prior_generator,
                dataset=dataset,
                fid_model=fid_model,
                fid_size=299,
                reals_batch_size=64
            ),
            interval=args.fid_interval
        )
        trainer.register_metric(
            name='FID',
            eval_fn=fid.FID(
                trainer.Gs,
                trainer.prior_generator,
                dataset=dataset,
                fid_model=fid_model,
                fid_size=None
            ),
            interval=args.fid_interval
        )
    if args.ppl_interval:
        lpips_model = lpips.LPIPS_VGG16(
            pixel_min=args.pixel_min, pixel_max=args.pixel_max)
        crop = None
        if args.ppl_ffhq_crop:
            crop = ppl.PPL.FFHQ_CROP
        trainer.register_metric(
            name='PPL_end',
            eval_fn=ppl.PPL(
                trainer.Gs,
                trainer.prior_generator,
                full_sampling=False,
                crop=crop,
                lpips_model=lpips_model,
                lpips_size=256
            ),
            interval=args.ppl_interval
        )
        trainer.register_metric(
            name='PPL_full',
            eval_fn=ppl.PPL(
                trainer.Gs,
                trainer.prior_generator,
                full_sampling=True,
                crop=crop,
                lpips_model=lpips_model,
                lpips_size=256
            ),
            interval=args.ppl_interval
        )
    if args.tensorboard_image_interval:
        for static in [True, False]:
            for trunc in [0.5, 0.7, 1.0]:
                if static:
                    name = 'static'
                else:
                    name = 'random'
                name += '/trunc_{:.1f}'.format(trunc)
                trainer.add_tensorboard_image_logging(
                    name=name,
                    num_images=4,
                    interval=args.tensorboard_image_interval,
                    resize=args.tensorboard_image_size,
                    seed=1234567890 if static else None,
                    truncation_psi=trunc,
                    pixel_min=args.pixel_min,
                    pixel_max=args.pixel_max
                )
    return trainer

#----------------------------------------------------------------------------

def run(args):
    if not (args.checkpoint_dir or args.output):
            warnings.warn(
                'Neither an output path or checkpoint dir has been ' + \
                'given. Weights from this training run will never ' + \
                'be saved.'
            )
    if args.output:
            assert os.path.isdir(args.output) or not os.path.splitext(args.output)[-1], \
                '--output argument should specify a directory, not a file.'
    trainer = get_trainer(args)
    trainer.train(iterations=args.iterations)
    if not args.output:
        print('Saving models to {}'.format(args.output))
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        for model_name in ['G', 'D', 'Gs']:
            getattr(trainer, model_name).save(os.path.join(args.output_dir, model_name + '.pth'))


#----------------------------------------------------------------------------

if __name__ == '__main__':
    config_file="./configs/my_settings.yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    run(cfg)
