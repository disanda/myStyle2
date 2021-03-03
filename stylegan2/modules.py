import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def get_activation(activation):
    if isinstance(activation, nn.Module) or callable(activation):
        return activation, 1.
    if isinstance(activation, str):
        activation = activation.lower()
    if activation in [None, 'linear']:
        return nn.Identity(), 1.
    lrelu_strings = ('leaky', 'leakyrely', 'leaky_relu', 'leaky relu', 'lrelu')
    if activation.startswith(lrelu_strings):
        for l_s in lrelu_strings:
            activation = activation.replace(l_s, '')
        slope = ''.join(
            char for char in activation if char.isdigit() or char == '.')
        slope = float(slope) if slope else 0.01
        return nn.LeakyReLU(slope), np.sqrt(2)  # close enough to true gain
    elif activation.startswith('swish'):
        return Swish(affine=activation != 'swish'), np.sqrt(2)
    elif activation in ['relu']:
        return nn.ReLU(), np.sqrt(2)
    elif activation in ['elu']:
        return nn.ELU(), 1.
    elif activation in ['prelu']:
        return nn.PReLU(), np.sqrt(2)
    elif activation in ['rrelu', 'randomrelu']:
        return nn.RReLU(), np.sqrt(2)
    elif activation in ['selu']:
        return nn.SELU(), 1.
    elif activation in ['softplus']:
        return nn.Softplus(), 1
    elif activation in ['softsign']:
        return nn.Softsign(), 1  # unsure about this gain
    elif activation in ['sigmoid', 'logistic']:
        return nn.Sigmoid(), 1.
    elif activation in ['tanh']:
        return nn.Tanh(), 1.
    else:
        raise ValueError(
            'Activation "{}" not available.'.format(activation)
        )

class Swish(nn.Module): # 'Swish' non-linear activation function. https://arxiv.org/pdf/1710.05941.pdf
    def __init__(self, affine=False): # affine (bool): Multiply the input to sigmoid with a learnable scale. Default value is False.
        super(Swish, self).__init__()
        if affine:
            self.beta = nn.Parameter(torch.tensor([1.]))
        self.affine = affine
    def forward(self, input, *args, **kwargs):
        x = input
        if self.affine:
            x *= self.beta
        return x * torch.sigmoid(x)

def _get_weight_and_coef(shape, lr_mul=1, weight_scale=True, gain=1, fill=None):
    fan_in = np.prod(shape[1:]) #shape=[3,3] or [3] or [], prod元素相乘,若为空列表[]时，prod值为1
    he_std = gain / np.sqrt(fan_in) #后一维特征根方，方差缩小了
    if weight_scale:
        init_std = 1 / lr_mul
        runtime_coef = he_std * lr_mul
    else:
        init_std = he_std / lr_mul
        runtime_coef = lr_mul
    weight = torch.empty(*shape)
    if fill is None:
        weight.normal_(0, init_std)
    else:
        weight.fill_(fill)
    return nn.Parameter(weight), runtime_coef #得道一个kernel_size为shape的可训练weight,以及一个影响影子

def _apply_conv(input, *args, transpose=False, **kwargs):
    dim = input.dim() - 2
    conv_fn = getattr(F, 'conv{}{}d'.format('_transpose' if transpose else '', dim)) #dim可选1D,2D,3D
    return conv_fn(input=input, *args, **kwargs) #通过 getattr去F里面选对应的conv函数，默认的(列表)，可选的(字典)

def _setup_mod_weight_for_t_conv(weight, in_channels, out_channels):#Reshape a modulated conv weight for use with a transposed convolution.
    weight = weight.view(-1, out_channels, in_channels, *weight.size()[2:])#[BO]I*k -> BOI*k, [out,in,h,w]-> [-1,out,in,h,w]
    weight = weight.transpose(1, 2) #BOI*k -> BIO*k, ->[-1,in,out,h,w]
    weight = weight.reshape(-1,out_channels,*weight.size()[3:]) # BIO*k -> [BI]O*k, ->[in,out,h,w]
    return weight #reshaped_weight (torch.Tensor)

def _setup_filter_kernel(filter_kernel=0, gain=1, up_factor=1, dim=2):
    filter_kernel = filter_kernel or 2 #如果为0或None,即false，则赋值为2, 否则是一个整数,或者矩阵
    if isinstance(filter_kernel, (int, float)): #如果为一个整数或浮点数
        def binomial(n, k):
            if k in [1, n]:
                return 1
            return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))#factorial->求阶乘
        filter_kernel = [binomial(filter_kernel, k) for k in range(1, filter_kernel + 1)] 
    if not torch.is_tensor(filter_kernel):
        filter_kernel = torch.tensor(filter_kernel)
    filter_kernel = filter_kernel.float()
    if filter_kernel.dim() == 1:
        _filter_kernel = filter_kernel.unsqueeze(0) #shape: [n]->[1,n]
        while filter_kernel.dim() < dim:
            filter_kernel = torch.matmul(filter_kernel.unsqueeze(-1), _filter_kernel)#shape: [n,1]*[1,n]=[n,n]
    assert all(filter_kernel.size(0) == s for s in filter_kernel.size())
    filter_kernel /= filter_kernel.sum()
    filter_kernel *= gain * up_factor ** 2
    return filter_kernel.float() #通过阶乘，得到一个中心值最大的矩阵:[filter_kernel,filter_kernel],矩阵元素值的和为1

def _get_layer(layer_class, kwargs, wrap=False, noise=False):
    layer = layer_class(**kwargs)
    if wrap:
        if noise:
            layer = NoiseInjectionWrapper(layer)
        layer = BiasActivationWrapper(layer, **kwargs)
    return layer

class DenseLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 lr_mul=1, # Learning rate multiplier of the weight. Weights are scaled by this value. 
                 weight_scale=True, # Scale weights for equalized learning rate.
                 gain=1,
                 *args,
                 **kwargs):
        super(DenseLayer, self).__init__()
        weight, weight_coef = _get_weight_and_coef(shape=[out_features, in_features], lr_mul=lr_mul, weight_scale=weight_scale, gain=gain)
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef
    def forward(self, input, **kwargs): #Perform a matrix multiplication of the weightof this layer and the input.
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        return input.matmul(weight.t()) #这个fc是一种特殊的fc，完成的是输入和weight矩阵的matmul

class BiasActivationWrapper(nn.Module):
    def __init__(self,
                 layer,
                 features=None,#The number of features of the output of the `layer`.
                 use_bias=True,
                 activation='linear',
                 bias_init=0,
                 lr_mul=1,#Learning rate multiplier of the bias weight.
                 weight_scale=True,#weight_scale (float): Scale weights for equalized learning rate.
                 *args,
                 **kwargs):
        super(BiasActivationWrapper, self).__init__()
        self.layer = layer
        bias = None
        bias_coef = None
        if use_bias:
            assert features, '`features` is required when using bias.'
            bias, bias_coef = _get_weight_and_coef(shape=[features],lr_mul=lr_mul, weight_scale=False, fill=bias_init)
        self.register_parameter('bias', bias)
        self.bias_coef = bias_coef
        self.act, self.gain = get_activation(activation)
    def forward(self, *args, **kwargs):
        x = self.layer(*args, **kwargs) # dense输出 
        if self.bias is not None:
            bias = self.bias.view(1, -1, *[1] * (x.dim() - 2)) #[1,-1,x.dim()-2]
            if self.bias_coef != 1:
                bias = self.bias_coef * bias
            x += bias
        x = self.act(x)
        if self.gain != 1:
            x *= self.gain
        return x # 这个函数后来被用于modulate, layer传入的是fc, 让fc再套一个activation

class NoiseInjectionWrapper(nn.Module):
    def __init__(self, layer, same_over_batch=True):
        super(NoiseInjectionWrapper, self).__init__()
        self.layer = layer
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('noise_storage', None)
        self.same_over_batch = same_over_batch
        self.random_noise()

    def has_noise_shape(self):
        return self.noise_storage is not None

    def random_noise(self):#Randomize noise for each new output.
        self._fixed_noise = False
        if isinstance(self.noise_storage, nn.Parameter):
            noise_storage = self.noise_storage
            del self.noise_storage
            self.register_buffer('noise_storage', noise_storage.data)

    def static_noise(self, trainable=False, noise_tensor=None):
        assert self.has_noise_shape(), \
            'Noise shape is unknown'
        if noise_tensor is None:
            noise_tensor = self.noise_storage
        else:
            noise_tensor = noise_tensor.to(self.weight)
        if trainable and not isinstance(noise_tensor, nn.Parameter):
            noise_tensor = nn.Parameter(noise_tensor)
        if isinstance(self.noise_storage, nn.Parameter) and not trainable:
            del self.noise_storage
            self.register_buffer('noise_storage', noise_tensor)
        else:
            self.noise_storage = noise_tensor
        self._fixed_noise = True
        return noise_tensor

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if name != 'noise_storage' and param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if name != 'noise_storage' and buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'noise_storage'
        if key in state_dict: del state_dict[key]
        return super(NoiseInjectionWrapper, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self): return 'static_noise={}'.format(self._fixed_noise)

    def forward(self, *args, **kwargs): #对运算层加噪音
        x = self.layer(*args, **kwargs)
        noise_shape = list(x.size())
        noise_shape[1] = 1
        if self.same_over_batch:
            noise_shape[0] = 1
        if self.noise_storage is None or list(self.noise_storage.size()) != noise_shape:
            if not self._fixed_noise:
                self.noise_storage = torch.empty(*noise_shape, dtype=self.weight.dtype,device=self.weight.device)
            else:
                assert list(self.noise_storage.size()[2:]) == noise_shape[2:], \
                    'A data size {} has been encountered, '.format(x.size()[2:]) + \
                    'the static noise previously set up does ' + \
                    'not match this size {}'.format(self.noise_storage.size()[2:])
                assert self.noise_storage.size(0) == 1 or self.noise_storage.size(0) == x.size(0), \
                    'Static noise batch size mismatch! ' + \
                    'Noise batch size: {}, '.format(self.noise_storage.size(0)) + \
                    'input batch size: {}'.format(x.size(0))
                assert self.noise_storage.size(1) == 1 or self.noise_storage.size(1) == x.size(1), \
                    'Static noise channel size mismatch! ' + \
                    'Noise channel size: {}, '.format(self.noise_storage.size(1)) + \
                    'input channel size: {}'.format(x.size(1))
        if not self._fixed_noise:  self.noise_storage.normal_()
        x += self.weight * self.noise_storage #weight是可以训练的，初始值为0,是噪声的系数加上正常的输出x
        return x

class FilterLayer(nn.Module):
    def __init__(self,
                 filter_kernel, # dims * (k,)-> [batch_size,channel] * kernel_size
                 stride=1,
                 pad0=0, #Amount to pad start of each data dimension. 
                 pad1=0, #Amount to pad end of each data dimension.
                 pad_mode='constant', #The constant value to pad
                 pad_constant=0,
                 *args,
                 **kwargs):
        super(FilterLayer, self).__init__()
        dim = filter_kernel.dim()
        filter_kernel = filter_kernel.view(1, 1, *filter_kernel.size())
        self.register_buffer('filter_kernel', filter_kernel)
        self.stride = stride
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
            self.pad_mode = pad_mode
            self.pad_constant = pad_constant

    def extra_repr(self):return 'filter_size={}, stride={}'.format(tuple(self.filter_kernel.size()[2:]), self.stride)

    def forward(self, input, **kwargs):
        x = input
        conv_kwargs = dict(
            weight=self.filter_kernel.repeat(input.size(1), *[1] * (self.filter_kernel.dim() - 1)),
            stride=self.stride,
            groups=input.size(1),
        )
        if self.fused_pad:
            conv_kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, transpose=False, **conv_kwargs)  #卷积操作的细分：和源码(F.conv)类似，先pad，再weight

class Upsample(nn.Module):
    def __init__(self,
                 mode='FIR', # pass to torch.nn.functional.interpolate(). FIR滤波器
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 gain=1,
                 dim=2, # Dims of data, excluding batch and channel dimensions
                 *args,
                 **kwargs):
        super(Upsample, self).__init__()
        assert mode != 'max', 'mode \'max\' can only be used for downsampling.'
        if mode == 'FIR': 
            if filter is None: 
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=gain, up_factor=2, dim=dim) # shape:[4,4]
            pad = filter_kernel.size(-1) - 1 # 3
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                pad0=(pad + 1) // 2 + 1, # 3
                pad1=pad // 2, # 1
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant)
            self.register_buffer('weight', torch.ones(*[1 for _ in range(dim + 2)]))
        self.mode = mode

    def extra_repr(self): return 'resample_mode={}'.format(self.mode)

    def forward(self, input, **kwargs):
        if self.mode == 'FIR':
            x = _apply_conv(
                input=input,
                weight=self.weight.expand(input.size(1), *self.weight.size()[1:]),
                groups=input.size(1),
                stride=2,
                transpose=True
            )
            x = self.filter(x) # 常量填充
        else:
            interp_kwargs = dict(scale_factor=2, mode=self.mode)
            if 'linear' in self.mode or 'cubic' in self.mode: interp_kwargs.update(align_corners=False)
            x = F.interpolate(input, **interp_kwargs)
        return x # 注意这个类并没有要学习的参数

class Downsample(nn.Module):
    def __init__(self,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs):
        super(Downsample, self).__init__()
        if mode == 'FIR':
            if filter is None: filter = [1, 1]
            filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=gain, up_factor=1, dim=dim)
            pad = filter_kernel.size(-1) - 2 # 4 - 2
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                stride=2, #下采样需要步幅为2
                pad0=pad // 2, # 1
                pad1=pad - pad // 2,#1 
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant)
        self.mode = mode

    def extra_repr(self): return 'resample_mode={}'.format(self.mode)

    def forward(self, input, **kwargs):
        if self.mode == 'FIR':
            x = self.filter(input)
        elif self.mode == 'max':
            return getattr(F, 'max_pool{}d'.format(input.dim() - 2))(input)
        else:
            x = F.interpolate(input, scale_factor=0.5, mode=self.mode)
        return x

class MinibatchStd(nn.Module):
    def __init__(self, group_size=4, eps=1e-8, *args, **kwargs):
        super(MinibatchStd, self).__init__()
        if group_size is None or group_size <= 0: 
            group_size = 0 # If <= 0, the entire batch is used. Default value is 4.
        assert group_size != 1, 'Can not use 1 as minibatch std group size.'
        self.group_size = group_size
        self.eps = eps # Epsilon value added for numerical stability. Default value is 1e-8.

    def extra_repr(self): return 'group_size={}'.format(self.group_size or '-1')

    def forward(self, input, **kwargs): #Add a new feature map to the input containing the average standard deviation for each slice.
        group_size = self.group_size or input.size(0)
        assert input.size(0) >= group_size, \
            'Can not use a smaller batch size ' + \
            '({}) than the specified '.format(input.size(0)) + \
            'group size ({}) '.format(group_size) + \
            'of this minibatch std layer.'
        assert input.size(0) % group_size == 0, \
            'Can not use a batch of a size ' + \
            '({}) that is not '.format(input.size(0)) + \
            'evenly divisible by the group size ({})'.format(group_size)
        x = input # B = batch size, C = num channels ,*s = the size dimensions (height, width for images)
        y = input.view(group_size, -1, *input.size()[1:]) # B,C,*s -> G[B/G]C*s
        y = y.float() # For numerical stability when training with mixed precision
        y -= y.mean(dim=0, keepdim=True) # G[B/G]C*s, 减去group的均值
        y = torch.mean(y ** 2, dim=0) # [B/G]C*s
        y = torch.sqrt(y + self.eps) # [B/G]C*s
        y = torch.mean(y.view(y.size(0), -1), dim=-1) # [B/G]
        y = y.view(-1, *[1] * (input.dim() - 1)) ## [B/G]1*1
        y = y.to(x) #Cast back to input dtype,  让y的类型和x一样
        y = y.repeat(group_size, *[1] * (y.dim() - 1)) # B1*1
        y = y.expand(y.size(0), 1, *x.size()[2:]) # B1*s
        x = torch.cat([x, y], dim=1) # B[C+1]*s
        return x # x运算过后多一个channel


class ConvLayer(nn.Module): #modulated (style mod), normalize. by modifying convolutional kernel weight and employing grouped convolutions for efficiency.
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size=None, # Only required when `modulate=True`
                 modulate=False, #  Applies a "style" given by a latent vector ,A dense layer is added that projects latent into scales for the data channels.
                 demodulate=False,# Normalize std of outputs.
                 kernel_size=3,
                 pad_mode='constant',
                 pad_constant=0,
                 lr_mul=1, # Learning rate multiplier of the weight. Weights are scaled by this value. Default value is 1.
                 weight_scale=True, #Scale weights for equalized learning rate. Default value is True.
                 gain=1,
                 dim=2, #Dims of data (excluding batch and channel dimensions). Default value is 2.
                 eps=1e-8, #Epsilon value added for numerical stability.
                 *args,
                 **kwargs):
        super(ConvLayer, self).__init__()
        assert modulate or not demodulate, '`demodulate=True` can only be used when `modulate=True`'
        if modulate: assert latent_size is not None, 'When using `modulate=True`, `latent_size` has to be specified.'
        kernel_shape = [out_channels, in_channels] + dim * [kernel_size]
        weight, weight_coef = _get_weight_and_coef(shape=kernel_shape, lr_mul=lr_mul, weight_scale=weight_scale, gain=gain)
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef
        if modulate:
            self.dense = BiasActivationWrapper(
                layer=DenseLayer(in_features=latent_size, out_features=in_channels, lr_mul=lr_mul, weight_scale=weight_scale, gain=1),
                features=in_channels,
                use_bias=True,
                activation='linear',
                bias_init=1,
                lr_mul=lr_mul,
                weight_scale=weight_scale)
        self.dense_reshape = [-1, 1, in_channels] + dim * [1] # [-1, 1, in_c, 1, 1]
        self.dmod_reshape = [-1, out_channels, 1] + dim * [1] # [-1, out_c, 1, 1, 1]
        pad = (kernel_size - 1) # 2
        pad0 = pad - pad // 2 # 1
        pad1 = pad - pad0 # 1
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
        self.pad_mode = pad_mode
        self.pad_constant = pad_constant
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.modulate = modulate
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.lr_mul = lr_mul
        self.weight_scale = weight_scale
        self.gain = gain
        self.dim = dim
        self.eps = eps

    def forward_mod(self, input, latent, weight, **kwargs):
        assert latent is not None, 'A latent vector is required for the forwad pass of a modulated conv layer.'
        style_mod = self.dense(input=latent) # B,I . B = batch size, I = num input channels
        style_mod = style_mod.view(*self.dense_reshape) # B,1,I,*1. *1 = multiple dimensions of size 1, with number of dimensions depending on data format.
        weight = weight.unsqueeze(0) # 1,O,I,*k. O = num output channels, *k = sizes of the convolutional kernel excluding in and out channel dimensions.
        weight = weight * style_mod # (1,O,I,*k)x(B,1,I,*1) -> B,O,I,*k
        if self.demodulate:
            dmod = torch.rsqrt(torch.sum(weight.view(weight.size(0),weight.size(1),-1) ** 2,dim=-1) + self.eps) # B,O
            dmod = dmod.view(*self.dmod_reshape) # B,O,1,*1
            weight = weight * dmod # (BOI*k)x(BO1*1) -> BOI*k
        x = input.view(1, -1, *input.size()[2:]) # B,I,*s -> 1,[BI],*s. *s = the size dimensions, example: (height, width) for images
        weight = weight.view(-1, *weight.size()[2:]) # BOI*k -> [BO]I*k
        x = self._process(input=x, weight=weight, groups=input.size(0)) # 1,[BO],*s
        x = x.view(-1, self.out_channels, *x.size()[2:]) # 1[BO]*s -> B,O,*s
        return x

    def forward(self, input, latent=None, **kwargs):
        weight = self.weight
        if self.weight_coef != 1: weight = self.weight_coef * weight
        if self.modulate: return self.forward_mod(input=input, latent=latent, weight=weight)
        return self._process(input=input, weight=weight)

    def _process(self, input, weight, **kwargs): #Pad input and convolve it returning the result.
        x = input
        if self.fused_pad: 
            kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, weight=weight, transpose=False, **kwargs)

    def extra_repr(self):
        string = 'in_channels={}, out_channels={}'.format(self.weight.size(1), self.weight.size(0))
        string += ', modulate={}, demodulate={}'.format(self.modulate, self.demodulate)
        return string


class ConvUpLayer(ConvLayer):#A convolutional upsampling layer that doubles the size of inputs.
    def __init__(self,
                 *args,
                 fused=True, # Fuse the upsampling operation with the convolution
                 mode='FIR', # Resample mode, can only be 'FIR' or 'none' if the operation is fused, otherwise it can also be one of the valid modes that can be passed to torch.nn.functional.interpolate().
                 filter=[1, 3, 3, 1], # Filter to use if `mode='FIR'`.
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True, # If FIR filter is used, do all the padding for both convolution and FIR in the FIR layer instead of once per layer.
                 **kwargs):
        super(ConvUpLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], 'Fused conv upsample can only use \'FIR\' or \'none\' for resampling (`mode` argument).'
            self.padding = np.ceil(self.kernel_size / 2 - 1) #向上取整
            self.output_padding = 2 * (self.padding + 1) - self.kernel_size
            if not self.modulate: # pre-prepare weights only once instead of every forward pass
                self.weight = nn.Parameter(self.weight.data.transpose(0, 1).contiguous()) #contiguous 保证数据连续
            self.filter = None
            if mode == 'FIR': 
                filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=self.gain, up_factor=2,dim=self.dim)
                if pad_once:
                    self.padding = 0
                    self.output_padding = 0
                    pad = (filter_kernel.size(-1) - 2) - (self.kernel_size - 1)
                    pad0 = (pad + 1) // 2 + 1,
                    pad1 = pad // 2 + 1,
                else:
                    pad = (filter_kernel.size(-1) - 1)
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(
                    filter_kernel=filter_kernel,
                    pad0=pad0,
                    pad1=pad1,
                    pad_mode=filter_pad_mode,
                    pad_constant=filter_pad_constant)
        else:
            assert mode != 'none', '\'none\' can not be used as ' + \
                'sampling `mode` when `fused=False` as upsampling ' + \
                'has to be performed separately from the conv layer.'
            self.upsample = Upsample(
                mode=mode,
                filter=filter,
                filter_pad_mode=filter_pad_mode,
                filter_pad_constant=filter_pad_constant,
                channels=self.in_channels,
                gain=self.gain,
                dim=self.dim
            )
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        x = input
        if self.fused:
            if self.modulate:
                weight = _setup_mod_weight_for_t_conv(weight=weight, in_channels=self.in_channels,out_channels=self.out_channels)
            pad_out = False
            if self.pad_mode == 'constant' and self.pad_constant == 0:
                if self.filter is not None or not self.pad_once:
                    kwargs.update(padding=self.padding, output_padding=self.output_padding)
            elif self.filter is None:
                if self.padding:
                    x = F.pad(x, [self.padding] * 2 * self.dim, mode=self.pad_mode, value=self.pad_constant)
                pad_out = self.output_padding != 0
            kwargs.update(stride=2)
            x = _apply_conv(input=x, weight=weight, transpose=True, **kwargs)
            if pad_out:
                x = F.pad(x, [self.output_padding, 0] * self.dim, mode=self.pad_mode, value=self.pad_constant)
            if self.filter is not None:
                x = self.filter(x)
        else:
            x = super(ConvUpLayer, self)._process(input=self.upsample(input=x), weight=weight, **kwargs)
        return x

    def extra_repr(self):
        string = super(ConvUpLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(self.fused, self.mode)
        return string


class ConvDownLayer(ConvLayer):#A convolutional downsampling layer that halves the size of inputs.
    def __init__(self,
                 *args,
                 fused=True, # Fuse the downsampling operation with the convolution, turning this layer into a strided convolution.
                 mode='FIR', # it can also be 'max' or one of the valid modes that can be passed to torch.nn.functional.interpolate().
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True, # If FIR filter is used, do all the padding for both convolution and FIR in the FIR layer instead of once per layer.
                 **kwargs):
        super(ConvDownLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], 'Fused conv downsample can only use \'FIR\' or \'none\' for resampling (`mode` argument).'
            pad = self.kernel_size - 2 # 3-2 = 1
            pad0 = pad // 2 # 0
            pad1 = pad - pad0 # 1
            if pad0 == pad1 and (pad0 == 0 or self.pad_mode == 'constant' and self.pad_constant == 0):
                self.fused_pad = True
                self.padding = pad0
            else:
                self.fused_pad = False
                self.padding = [pad0, pad1] * self.dim
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(filter_kernel=filter, gain=self.gain, up_factor=1,dim=self.dim)
                if pad_once:
                    self.fused_pad = True
                    self.padding = 0
                    pad = (filter_kernel.size(-1) - 2) + (self.kernel_size - 1)
                    pad0 = (pad + 1) // 2,
                    pad1 = pad // 2,
                else:
                    pad = (filter_kernel.size(-1) - 1)
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(
                    filter_kernel=filter_kernel,
                    pad0=pad0,
                    pad1=pad1,
                    pad_mode=filter_pad_mode,
                    pad_constant=filter_pad_constant
                )
                self.pad_once = pad_once
        else:
            assert mode != 'none', '\'none\' can not be used as ' + \
                'sampling `mode` when `fused=False` as downsampling ' + \
                'has to be performed separately from the conv layer.'
            self.downsample = Downsample(
                mode=mode,
                filter=filter,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant,
                channels=self.in_channels,
                gain=self.gain,
                dim=self.dim
            )
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        x = input
        if self.fused:
            kwargs.update(stride=2)
            if self.filter is not None:
                x = self.filter(input=x)
        else:
            x = self.downsample(input=x)
        x = super(ConvDownLayer, self)._process(input=x, weight=weight, **kwargs)
        return x

    def extra_repr(self):
        string = super(ConvDownLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(self.fused, self.mode)
        return string


class GeneratorConvBlock(nn.Module): # A convblock for the synthesiser model.
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size,
                 demodulate=True, # Normalize feature outputs from conv layers.
                 resnet=False,
                 up=False, # Upsample the data to twice its size. This is performed in the first layer of the block.
                 num_layers=2, # Number of convolutional layers of this block. Default value is 2.
                 filter=[1, 3, 3, 1],
                 activation='leaky:0.2',
                 mode='FIR',
                 fused=True, # If `up=True`, fuse the upsample operation and the first convolutional layer into a transposed convolutional layer.
                 kernel_size=3,
                 pad_mode='constant',
                 pad_constant=0,
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True,
                 use_bias=True,
                 noise=True, # Add noise to the output of each layer. 
                 lr_mul=1, # The learning rate multiplier for this block.
                 weight_scale=True, # Use weight scaling for equalized learning rate. 
                 gain=1,
                 dim=2,
                 eps=1e-8, # Epsilon value added for numerical stability.
                 *args,
                 **kwargs):
        super(GeneratorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(features=out_channels, modulate=True,)

        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if up:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, \
                '`mode` {} '.format(mode) + \
                'is not one of the available sample modes {}.'.format(available_sampling)

        self.conv_block = nn.ModuleList()

        while len(self.conv_block) < num_layers:
            use_up = up and not self.conv_block #第一个块有点不一样，conv需要up一下
            self.conv_block.append(_get_layer(ConvUpLayer if use_up else ConvLayer, layer_kwargs, wrap=True, noise=noise))
            layer_kwargs.update(in_channels=out_channels)

        self.projection = None
        if resnet:
            projection_kwargs = {**layer_kwargs, 'in_channels': in_channels, 'kernel_size': 1, 'modulate': False, 'demodulate': False}
            self.projection = _get_layer(ConvUpLayer if up else ConvLayer, projection_kwargs, wrap=False)
        self.res_scale = 1 / np.sqrt(2) # resnet有两个特征，需要压缩比例

    def __len__(self): return len(self.conv_block)# Get the number of conv layers in this block.

    def forward(self, input, latents=None, **kwargs):
        if latents.dim() == 2:
            latents.unsqueeze(1)
        if latents.size(1) == 1:
            latents = latents.repeat(1, len(self), 1)
        assert latents.size(1) == len(self), \
            'Number of latent inputs ({}) does not match '.format(latents.size(1)) + \
            'number of conv layers ({}) in block.'.format(len(self))
        x = input
        for i, layer in enumerate(self.conv_block):
            x = layer(input=x, latent=latents[:, i])
        if self.projection is not None: # residual
            x += self.projection(input=input)
            x *= self.res_scale
        return x

class DiscriminatorConvBlock(nn.Module): # A convblock for the discriminator model.
    def __init__(self,
                 in_channels,
                 out_channels,
                 resnet=False,
                 down=False, # Downsample the data to twice its size. This is performed in the last layer of the block.
                 num_layers=2, # 一个block块能的conv数量
                 filter=[1, 3, 3, 1],
                 activation='leaky:0.2',
                 mode='FIR',
                 fused=True, # fused (bool): If `down=True`, fuse the downsample operation and the last convolutional layer into a strided convolutional layer.
                 kernel_size=3,
                 pad_mode='constant', # Has to be one of 'constant', 'reflect', 'replicate' or 'circular'.
                 pad_constant=0,
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True,
                 use_bias=True,
                 lr_mul=1,
                 weight_scale=True,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs,#forward的输入 
                 ):
        super(DiscriminatorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(out_channels=in_channels, features=in_channels, modulate=False, demodulate=False)

        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if down:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('max')
                available_sampling.append('area')
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, \
                '`mode` {} '.format(mode) + \
                'is not one of the available sample ' + \
                'modes {}'.format(available_sampling)

        self.conv_block = nn.ModuleList()

        while len(self.conv_block) < num_layers:
            if len(self.conv_block) == num_layers - 1:
                layer_kwargs.update(out_channels=out_channels, features=out_channels)
            use_down = down and len(self.conv_block) == num_layers - 1
            self.conv_block.append(_get_layer(ConvDownLayer if use_down else ConvLayer, layer_kwargs, wrap=True, noise=False))

        self.projection = None
        if resnet:
            projection_kwargs = {**layer_kwargs, 'in_channels': in_channels, 'kernel_size': 1, 'modulate': False, 'demodulate': False}
            self.projection = _get_layer(ConvDownLayer if down else ConvLayer, projection_kwargs, wrap=False)
        self.res_scale = 1 / np.sqrt(2)

        self.in_channels = in_channels
        self.output_channels = out_channels

    def __len__(self): return len(self.conv_block) # Get the number of conv layers in this block.

    def forward(self, input, **kwargs):
        x = input
        for layer in self.conv_block:
            x = layer(input=x)
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x
