from yacs.config import CfgNode as CN


args = CN()


args.channels= [32, 64, 128, 256, 512, 512, 512, 512] # 9 -> 1024
args.gpu= [0]
args.output= 'output'  # Output directory for model weights.


# model
args.latent = 512 # Size of the prior (noise vector)
args.label = 0 # Number of unique labels. Unused if not specified.
args.base_shape = (4, 4) # Data shape of first layer in generator or last layer in discriminator.'
args.kernel_size = 3
args.pad_once = True # Pad filtered convs only once before filter instead of twice.
args.pad_mode = 'constant' # adding mode for conv layers
args.pad_constant = 0 # Padding constant for conv layers when `pad_mode` is 'constant'
args.filter_pad_mode = 'constant' # Padding mode for filter layers 
args.filter_pad_constant = 0 # Padding constant for filter layers when `filter_pad_mode` is 'constant'
args.filter = [1, 3, 3, 1] # 'Filter to use whenever FIR is applied.'
args.weight_scale = True # Use weight scaling for equalized learning rate

# Generator
args.g_file = None # Load a generator model from a file instead of constructing a new one. Disabled unless a file is specified.
args.g_channels = [] # Instead of the values of "--channels", use these for the generator instead.
args.g_skip = True # Use skip connections for the generator.
args.g_resnet = False # Use resnet connections for the generator.
args.g_conv_block_size = 2 # Number of layers in a conv block in the generator.
args.g_normalize = True # Normalize conv features for generator.
args.g_fused_conv = True # Fuse conv & upsample into a transposed conv for the generator.
args.g_activation = 'leaky=0.2' # The non-linear activaiton function for the generator.
args.g_conv_resample_mode = 'FIR' # Resample mode for upsampling conv layers for generator.
args.g_skip_resample_mode = 'FIR' # Resample mode for skip connection upsamples for the generator.
args.g_lr = 2e-3 # The learning rate for the generator
args.g_betas = (0, 0.99) # Beta values for the generator Adam optimizer.
args.g_loss = 'logistic_ns' # Loss function for the generator.
args.g_reg = 'pathreg=2' # Regularization function for the generator with an optional weight (=?).
args.g_reg_interval = 4 # Interval at which to regularize the generator.
args.g_iter = 1 # Number of generator iterations per training iteration.
args.style_mix = 0.9 # The probability of passing more than one latent to the generator.
args.latent_mapping_layers = 8 # The number of layers of the latent mapping network.
args.latent_mapping_lr_mul = 0.01 # The learning rate multiplier for the latent mapping network.
args.normalize_latent = True # Normalize latent inputs
args.modulate_rgb = True #Modulate RGB layers (use style for output layers of generator).

# Discriminator options
args.d_file = None
args.d_channels = [] 
args.d_skip = False 
args.d_resnet = True # here D use !
args.d_conv_block_size = 2 
args.d_fused_conv = True # D no normalize !
args.group_size = 4 # Size of the groups in batch std layer.
args.d_activation = 'leaky=0.2'
args.d_conv_resample_mode = 'FIR'
args.d_skip_resample_mode = 'FIR'
args.d_lr = 2e-3
args.d_betas = (0, 0.99)
args.d_loss = 'logistic'
args.d_reg = 'r1=10'
args.d_reg_interval = 16 # Interval at which to regularize the discriminator.
args.d_iter = 1

# Training options
args.iterations = 1000000 
args.gpu = [] # The cuda device(s) to use. Example= ""--gpu 0 1" will train on GPU 0 and GPU 1. Default= Only use CPU'
args.batch_size = 32
args.half = False # Use mixed precision training.
args.resume = False #Resume from the latest saved checkpoint in the checkpoint_dir. This loads all previous training settings except for the dataset options

# Extra metric options
args.fid_interval = None # type=int ,If specified, evaluate the FID metric with this interval.
args.ppl_interval = None # If specified, evaluate the PPL metric with this interval.
args.ppl_ffhq_crop = False # Crop images evaluated for PPL with crop values for FFHQ.

# Data options
args.pixel_min = -1 # type = float, Minimum of the value range of pixels in generated images.
args.pixel_max = 1 
args.data_channels = 3
args.data_dir = None
args.data_resize = True # type= bool ,Resize data to fit input size of discriminator. 
args.mirror_augment = False # Use random horizontal flipping for data images.
args.data_workers = 4 # Number of worker processes that handles dataloading.

# Logging options
args.checkpoint_dir = None
args.checkpoint_interval = 5
args.tensorboard_log_dir = None # type= str
args.tensorboard_image_interval = None # type= int
args.tensorboard_image_size = 256 # Size of images logged to tensorboard.


def get_cfg_defaults():
  return args.clone()














