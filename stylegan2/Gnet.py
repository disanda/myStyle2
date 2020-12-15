class GeneratorSynthesis(_BaseAdverserialModel):
    def __init__(self, **kwargs):
        super(GeneratorSynthesis, self).__init__()
        self._update_default_kwargs(
            latent_size=512,
            demodulate=True,
            modulate_data_out=True,
            noise=True,
            resnet=False,
            skip=True,
            const = None,
        )
        self._update_kwargs(**kwargs)

        if const == None:
            self.const = torch.nn.Parameter(torch.empty(self.channels[-1], *self.base_shape).normal_())
        else:
            self.const = const

        conv_block_kwargs = dict(
            latent_size=self.latent_size,
            demodulate=self.demodulate,
            resnet=self.resnet,
            up=True,
            num_layers=self.conv_block_size,
            filter=self.conv_filter,
            activation=self.activation,
            mode=self.conv_resample_mode,
            fused=self.fused_resample,
            kernel_size=self.kernel_size,
            pad_mode=self.conv_pad_mode,
            pad_constant=self.conv_pad_constant,
            filter_pad_mode=self.filter_pad_mode,
            filter_pad_constant=self.filter_pad_constant,
            pad_once=self.pad_once,
            noise=self.noise,
            lr_mul=self.lr_mul,
            weight_scale=self.weight_scale,
            gain=1,
            dim=self.dim,
            eps=self.eps
        )
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            modules.GeneratorConvBlock(
                **{
                    **conv_block_kwargs,
                    'in_channels': self.channels[-1],
                    'out_channels': self.channels[-1],
                    'resnet': False,
                    'up': False,
                    'num_layers': 1
                }
            )
        )
        for i in range(1, len(self.channels)):
            self.conv_blocks.append(
                modules.GeneratorConvBlock(
                    in_channels=self.channels[-i],
                    out_channels=self.channels[-i - 1],
                    **conv_block_kwargs
                )
            )
        self.to_data_layers = nn.ModuleList()
        for i in range(1, len(self.channels) + 1):
            to_data = None
            if i == len(self.channels) or self.skip:
                to_data = modules.BiasActivationWrapper(
                    layer=modules.ConvLayer(
                        **{
                            **conv_block_kwargs,
                            'in_channels': self.channels[-i],
                            'out_channels': self.data_channels,
                            'modulate': self.modulate_data_out,
                            'demodulate': False,
                            'kernel_size': 1
                        }
                    ),
                    **{
                        **conv_block_kwargs,
                        'features': self.data_channels,
                        'use_bias': True,
                        'activation': 'linear',
                        'bias_init': 0
                    }
                )
            self.to_data_layers.append(to_data)

        self.upsample = None
        if self.skip:
            self.upsample = modules.Upsample(
                mode=self.skip_resample_mode,
                filter=self.skip_filter,
                filter_pad_mode=self.filter_pad_mode,
                filter_pad_constant=self.filter_pad_constant,
                gain=1,
                dim=self.dim
            )

        self._num_latents = 1 + self.conv_block_size * (len(self.channels) - 1)
        if self.modulate_data_out:
            self._num_latents += 1

    def __len__(self):
        return self._num_latents

    def random_noise(self):
        for module in self.modules():
            if isinstance(module, modules.NoiseInjectionWrapper):
                module.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        rtn_tensors = []

        if not self.noise:
            return rtn_tensors

        for module in self.modules():
            if isinstance(module, modules.NoiseInjectionWrapper):
                has_noise_shape = module.has_noise_shape()
                device = module.weight.device
                dtype = module.weight.dtype
                break
        if not has_noise_shape:
            with torch.no_grad():
                self(torch.zeros(
                    1, len(self), self.latent_size, device=device, dtype=dtype))
        i = 0
        for block in self.conv_blocks:
            for layer in block.conv_block:
                for module in layer.modules():
                    if isinstance(module, modules.NoiseInjectionWrapper):
                        noise_tensor = None
                        if noise_tensors is not None:
                            if i < len(noise_tensors):
                                noise_tensor = noise_tensors[i]
                                i += 1
                            else:
                                rtn_tensors.append(None)
                                continue
                        rtn_tensors.append(
                            module.static_noise(trainable=trainable, noise_tensor=noise_tensor))
        if noise_tensors is not None:
            assert len(rtn_tensors) == len(noise_tensors), \
                'Got a list of {} '.format(len(noise_tensors)) + \
                'noise tensors but there are ' + \
                '{} noise layers in this model'.format(len(rtn_tensors))

        return rtn_tensors

    def forward(self, latents):
        assert latents.dim() == 3 and latents.size(1) == len(self), \
            'Input mismatch, expected latents of shape ' + \
            '(batch_size, {}, latent_size) '.format(len(self)) + \
            'but got {}.'.format(tuple(latents.size()))
        x = self.const.unsqueeze(0)
        y = None
        layer_idx = 0
        for block, to_data in zip(self.conv_blocks, self.to_data_layers):
            block_latents = latents[:, layer_idx:layer_idx + len(block)]
            x = block(input=x, latents=block_latents)
            layer_idx += len(block)
            if self.upsample is not None and layer_idx < len(self):
                if y is not None:
                    y = self.upsample(y)
            if to_data is not None:
                t = to_data(input=x, latent=latents[:, layer_idx])
                y = t if y is None else y + t
        return y