import warnings
import functools
import os
import time
import sys
import json
import numpy as np
import torch
import torch.utils.tensorboard
from torch import nn
import torchvision
try:
    import apex
    from apex import amp
except ImportError:
    pass

from . import models, utils, loss_fns

class Trainer:
    """
        dataset (indexable object): The dataset. Has to implement '__getitem__' and '__len__'. 
            If `label_size` > 0, this dataset object has to return both a data entry and its
            label when calling '__getitem__'.
        Gs (Generator, optional): A generator copy with the current
            moving average of the training generator. If not specified,
            a copy of the generator is made for the moving average of
            weights.
        Gs_device (str, int, torch.device, optional): The device to store
            the moving average weights on. If using a different device
            than what is specified for the `device` argument, updating
            the moving average weights will take longer as the data
            will have to be transfered over different devices. If
            this argument is not specified, the same device is used
            as specified in the `device` argument.
        G_reg (str, callable, None): The regularizer function to use
            for the generator. If string, it can only be 'pathreg'
            (pathlength regularization). A weight for the regularizer
            can be passed after the string name like the following:
                G_reg='pathreg:5'
            This will assign a weight of 5 to the regularization loss.
            If set to None, no geenerator regularization is performed.
            Default value is 'pathreg:2'.
        G_reg_interval (int): The interval at which to regularize the
            generator. If set to 0, the regularization and loss gradients
            are combined in a single optimization step every iteration.
            If set to 1, the gradients for the regularization and loss
            are used separately for two optimization steps. Any value
            higher than 1 indicates that regularization should only
            be performed at this interval (lazy regularization).
            Default value is 4.
        G_reg_batch_size (int): Same as `batch_size` but only for the regularization loss of the generator. Default value is 16.
        D_reg (str, callable, None): The regularizer function to use
            for the discriminator. If string, the following values
            can be used: 'r1', 'r2', 'gp'. See doc for `G_reg` for
            rest of info on regularizer format.
            Default value is 'r1:10'.
        D_reg_interval (int): Same as `D_reg_interval` but for the discriminator. Default value is 16.
        style_mix_prob (float): The probability of passing 2 latents instead of 1 to the generator during training. Default value is 0.9.
        pl_avg (float, torch.Tensor): The average pathlength starting value for pathlength regularization of the generator. Default value is 0.
        tensorboard_log_dir (str, optional): A path to a directory to log training values in for tensorboard. 
        checkpoint_dir (str, optional): A path to a directory to save training checkpoints to. If not specified, not checkpoints are automatically saved during training.
        checkpoint_interval (int): The interval at which to save training checkpoints. Default value is 10000.
        seen (int): The number of previously trained iterations. Used for logging. Default value is 0.
        half (bool): Use mixed precision training. Default value is False.
    """

    def __init__(self,
                 G, # G (Generator): The generator model.
                 D, # D (Discriminator): The discriminator model.
                 latent_size, # latent_size (int): The size of the latent inputs.
                 dataset,
                 Gs=None,
                 Gs_beta=0.5 ** (32 / 10000),#Gs_beta (float): The beta value for the moving average weights. Default value is 1 / (2 ^(32 / 10000)).
                 Gs_device=None,
                 batch_size=32,
                 label_size=0, # label_size (int, optional): Number of possible class labels. This is required for conditioning the GAN with labels.
                 data_workers=4,
                 G_loss='logistic_ns', #it can be one of the following: 'logistic', 'logistic_ns' or 'wgan'.(default is non-saturating logistic)
                 D_loss='logistic', # it can be one of the following: 'logistic' or 'wgan'.
                 G_reg='pathreg:2',
                 G_reg_interval=4,
                 G_opt_class='Adam', #  G_opt_class (str, class): The optimizer class for the generator. Default value is 'Adam'.
                 G_opt_kwargs={'lr': 2e-3, 'betas': (0, 0.99)}, #   G_opt_kwargs (dict): Keyword arguments for the generator optimizer constructor.
                 D_reg='r1:10',
                 D_reg_interval=16,
                 D_opt_class='Adam', #  D_opt_class (str, class): The optimizer class for the discriminator.
                 D_opt_kwargs={'lr': 2e-3, 'betas': (0, 0.99)},
                 style_mix_prob=0.9,
                 G_iter=1, # Number of generator iterations for every full training
                 D_iter=1, # Number of discriminator iterations for every full training
                 pl_avg=0.,
                 tensorboard_log_dir=None,
                 checkpoint_dir=None,
                 checkpoint_interval=10,
                 seen=0,
                 half=False,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = locals() # We store the training settings in a dict that can be saved as a json file.
        kwargs.pop('self') # First we remove the arguments that can not be turned into json.
        kwargs.pop('G')
        kwargs.pop('D')
        kwargs.pop('Gs')
        kwargs.pop('dataset') 
        kwargs.update(pl_avg=float(pl_avg)) # Some arguments may have to be turned into strings to be compatible with json.
        kwargs.update(device=str(self.device))
        if isinstance(Gs_device, torch.device): kwargs.update(device=str(Gs_device))
        self.kwargs = kwargs

        # Set up the models
        self.G = G.train().to(self.device)
        self.D = D.train().to(self.device)

        # Set up variable to keep track of moving average of path lengths
        self.pl_avg = torch.tensor(pl_avg, dtype=torch.float16 if half else torch.float32, device=self.device)

        # Set up moving average of generator
        self.Gs = Gs
        if not isinstance(Gs, utils.MovingAverageModule):
                self.Gs = utils.MovingAverageModule(
                    from_module=self.G,
                    to_module=Gs,
                    param_beta=Gs_beta,
                    device=self.device if Gs_device is None else Gs_device)

        # Set up loss and regularization functions
        self.G_loss = get_loss_fn('G', G_loss)
        self.D_loss = get_loss_fn('D', D_loss)
        self.G_reg = get_reg_fn('G', G_reg, pl_avg=self.pl_avg)
        self.D_reg = get_reg_fn('D', D_reg)
        self.G_reg_interval = G_reg_interval
        self.D_reg_interval = D_reg_interval
        self.G_iter = G_iter
        self.D_iter = D_iter
        self.batch_size =batch_size
        # Set up optimizers (adjust hyperparameters if lazy regularization is active)
        self.G_opt = build_opt(self.G, G_opt_class, G_opt_kwargs, self.G_reg, self.G_reg_interval)
        self.D_opt = build_opt(self.D, D_opt_class, D_opt_kwargs, self.D_reg, self.D_reg_interval)

        # Set up mixed precision training
        if half:
            assert 'apex' in sys.modules, 'Can not run mixed precision training (`half=True`) without the apex module.'
            (self.G, self.D), (self.G_opt, self.D_opt) = amp.initialize([self.G, self.D], [self.G_opt, self.D_opt], opt_level='O1')
        self.half = half

        # Data
        sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=data_workers,
            shuffle=sampler is None,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
        self.dataloader_iter = None
        self.prior_generator = utils.PriorGenerator(
            latent_size=latent_size,
            label_size=label_size, # None
            batch_size=self.batch_size,
            device=self.device
        )

        self.tb_writer = None
        if tensorboard_log_dir: self.tb_writer = torch.utils.tensorboard.SummaryWriter(tensorboard_log_dir)

        self.label_size = label_size
        self.style_mix_prob = style_mix_prob
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.seen = seen
        self.metrics = {}
        self.callbacks = []

    def _get_batch(self): #Fetch a batch and its labels. If no labels are available the returned labels will be `None`. 
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = None
            return self._get_batch()
        if isinstance(batch, (tuple, list)):
            if len(batch) > 1:
                data, label = batch[:2]
            else:
                data, label = batch[0], None
        else:
            data, label = batch, None
        if not self.label_size:
            label = None
        if torch.is_tensor(data):
            data = data.to(self.device)
        if torch.is_tensor(label):
            label = label.to(self.device)
        return data, label

    def _backward(self, loss, opt, mul=1, subdivisions=None):
        """
        Reduce loss by world size and subdivisions before
        calling backward for the loss. Loss scaling is
        performed when mixed precision training is
        enabled.
        Arguments:
            loss (torch.Tensor)
            opt (torch.optim.Optimizer)
            mul (float): Loss weight. Default value is 1.
            subdivisions (int, optional): The number of
                subdivisions to divide by. If this is
                not specified, the subdvisions from
                the specified batch and device size
                at construction is used.
        Returns:
            loss (torch.Tensor): The loss scaled by mul
                and subdivisions but not by world size.
        """
        if loss is None:
            return 0
        if self.half:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #get the scalar only
        return loss.item()

    def train(self, iterations, callbacks=None, verbose=True):
        """
        Train the models for a specific number of iterations.
        Arguments:
            iterations (int): Number of iterations to train for.
            callbacks (callable, list, optional): One
                or more callbacks to call at the end of each
                iteration. The function is given the total
                number of batches that have been processed since
                this trainer object was initialized (not reset when loading a saved checkpoint).
                Default value is None (unused).
            verbose (bool): Write progress to stdout. Default value is True.
        """
        evaluated_metrics = {}
        if verbose:
            progress = utils.ProgressWriter(iterations)
            value_tracker = utils.ValueTracker()
        for i in range(iterations):
            # Figure out if G and/or D be
            # regularized this iteration
            G_reg = self.G_reg is not None  #True
            if self.G_reg_interval and G_reg:
                G_reg = self.seen % self.G_reg_interval == 0
            D_reg = self.D_reg is not None #True
            if self.D_reg_interval and D_reg:
                D_reg = self.seen % self.D_reg_interval == 0

            # -----| Train G |----- #
            self.G.requires_grad_(True) # Disable gradients for D while training G, enable grads for G
            self.D.requires_grad_(False)

            for _ in range(self.G_iter):
                self.G_opt.zero_grad()
                G_loss = 0
                latents, latent_labels = self.prior_generator(multi_latent_prob=self.style_mix_prob)
                loss, _ = self.G_loss(G=self.G, D=self.D, latents=latents, latent_labels=latent_labels)
                G_loss += self._backward(loss, self.G_opt)

                if G_reg:
                    if self.G_reg_interval:
                        # For lazy regularization, even if the interval is set to 1, the optimization step is taken
                        # before the gradients of the regularization is gathered.
                        self.G_opt.step()
                        self.G_opt.zero_grad()
                    G_reg_loss = 0
                    latents, latent_labels = self.prior_generator(batch_size=self.batch_size, multi_latent_prob=self.style_mix_prob)
                    _, reg_loss = self.G_reg(G=self.G, latents=latents, latent_labels=latent_labels)
                    G_reg_loss += self._backward(reg_loss, self.G_opt, mul=self.G_reg_interval or 1)
                self.G_opt.step() # Update moving average of weights after, each G training subiteration
                if self.Gs is not None:
                    self.Gs.update()

            # -----| Train D |----- #Re-enable gradients for D && Disable gradients for G while training D
            self.D.requires_grad_(True)
            self.G.requires_grad_(False)

            for _ in range(self.D_iter):
                self.D_opt.zero_grad()

                D_loss = 0
                latents, latent_labels = self.prior_generator(multi_latent_prob=self.style_mix_prob)
                reals, real_labels = self._get_batch()
                loss, _ = self.D_loss(G=self.G, D=self.D, latents=latents, latent_labels=latent_labels, reals=reals, real_labels=real_labels)
                D_loss += self._backward(loss, self.D_opt)

                if D_reg:
                    if self.D_reg_interval:
                        # For lazy regularization, even if the interval is set to 1, the optimization step is taken
                        # before the gradients of the regularization is gathered.
                        self.D_opt.step()
                        self.D_opt.zero_grad()
                    D_reg_loss = 0
                    latents, latent_labels = self.prior_generator(multi_latent_prob=self.style_mix_prob)
                    reals, real_labels = self._get_batch()
                    _, reg_loss = self.D_reg(
                            G=self.G,
                            D=self.D,
                            latents=latents,
                            latent_labels=latent_labels,
                            reals=reals,
                            real_labels=real_labels)
                    D_reg_loss += self._backward(reg_loss, self.D_opt, mul=self.D_reg_interval or 1)
                self.D_opt.step()

            if self.tb_writer is not None or verbose:
                # In case verbose is true and tensorboard logging enabled
                # we calculate grad norm here to only do it once as well
                # as making sure we do it before any metrics that may possibly zero the grads.
                G_grad_norm = utils.get_grad_norm_from_optimizer(self.G_opt)
                D_grad_norm = utils.get_grad_norm_from_optimizer(self.D_opt)

            for name, metric in self.metrics.items():
                if not metric['interval'] or self.seen % metric['interval'] == 0:
                    evaluated_metrics[name] = metric['eval_fn']()

            # Printing and logging

            # Tensorboard logging
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('Loss/G_loss', G_loss, self.seen)
                if G_reg:
                    self.tb_writer.add_scalar('Loss/G_reg', G_reg_loss, self.seen)
                    self.tb_writer.add_scalar('Grad_norm/G_reg', G_grad_norm, self.seen)
                    self.tb_writer.add_scalar('Params/pl_avg', self.pl_avg, self.seen)
                else:
                    self.tb_writer.add_scalar('Grad_norm/G_loss', G_grad_norm, self.seen)
                self.tb_writer.add_scalar('Loss/D_loss', D_loss, self.seen)
                if D_reg:
                    self.tb_writer.add_scalar('Loss/D_reg', D_reg_loss, self.seen)
                    self.tb_writer.add_scalar('Grad_norm/D_reg', D_grad_norm, self.seen)
                else:
                    self.tb_writer.add_scalar('Grad_norm/D_loss', D_grad_norm, self.seen)
                for name, value in evaluated_metrics.items():
                    self.tb_writer.add_scalar('Metrics/{}'.format(name), value, self.seen)

            # Printing
            if verbose:
                value_tracker.add('seen', self.seen + 1, beta=0)
                value_tracker.add('G_lr', self.G_opt.param_groups[0]['lr'], beta=0)
                value_tracker.add('G_loss', G_loss)
                if G_reg:
                    value_tracker.add('G_reg', G_reg_loss)
                    value_tracker.add('G_reg_grad_norm', G_grad_norm)
                    value_tracker.add('pl_avg', self.pl_avg, beta=0)
                else:
                    value_tracker.add('G_loss_grad_norm', G_grad_norm)
                value_tracker.add('D_lr', self.D_opt.param_groups[0]['lr'], beta=0)
                value_tracker.add('D_loss', D_loss)
                if D_reg:
                    value_tracker.add('D_reg', D_reg_loss)
                    value_tracker.add('D_reg_grad_norm', D_grad_norm)
                else:
                    value_tracker.add('D_loss_grad_norm', D_grad_norm)
                for name, value in evaluated_metrics.items():
                    value_tracker.add(name, value, beta=0)
                progress.write(str(value_tracker))

            # Callback
            for callback in utils.to_list(callbacks) + self.callbacks:
                callback(self.seen)

            self.seen += 1
            
            # clear cache
            torch.cuda.empty_cache()
            # Handle checkpointing
            if self.checkpoint_dir and self.checkpoint_interval:
                if self.seen % self.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir,'{}_{}'.format(self.seen, time.strftime('%Y-%m-%d_%H-%M-%S')))
                    self.save_checkpoint(checkpoint_path)
        if verbose:
            progress.close()

    def register_metric(self, name, eval_fn, interval):
        """
        Add a metric. This will be evaluated every `interval`
        training iteration. Used by tensorboard and progress
        updates written to stdout while training.
        Arguments:
            name (str): A name for the metric. If a metric with
                this name already exists it will be overwritten.
            eval_fn (callable): A function that evaluates the metric
                and returns a python number.
            interval (int): The interval to evaluate at.
        """
        self.metrics[name] = {'eval_fn': eval_fn, 'interval': interval}

    def remove_metric(self, name):
        """
        Remove a metric that was previously registered.
        Arguments:
            name (str): Name of the metric.
        """
        if name in self.metrics:
            del self.metrics[name]
        else:
            warnings.warn(
                'Attempting to remove metric {} '.format(name) + \
                'which does not exist.'
            )

    def generate_images(self,
                        num_images,
                        seed=None,
                        truncation_psi=None,
                        truncation_cutoff=None,
                        label=None,
                        pixel_min=-1,
                        pixel_max=1):
        """
        Generate some images with the generator and transform them into PIL
        images and return them as a list.
        Arguments:
            num_images (int): Number of images to generate.
            seed (int, optional): The seed for the random generation
                of input latent values.
            truncation_psi (float): See stylegan2.model.Generator.set_truncation()
                Default value is None.
            truncation_cutoff (int): See stylegan2.model.Generator.set_truncation()
            label (int, list, optional): Label to condition all generated images with
                or multiple labels, one for each generated image.
            pixel_min (float): The min value in the pixel range of the generator.
                Default value is -1.
            pixel_min (float): The max value in the pixel range of the generator.
                Default value is 1.
        Returns:
            images (list): List of PIL images.
        """
        if seed is None:
            seed = int(10000 * time.time())
        latents, latent_labels = self.prior_generator(num_images, seed=seed)
        if label:
            assert latent_labels is not None, 'Can not specify label when no labels ' + \
                'are used by this model.'
            label = utils.to_list(label)
            assert all(isinstance(l, int) for l in label), '`label` can only consist of ' + \
                'one or more python integers.'
            assert len(label) == 1 or len(label) == num_images, '`label` can either ' + \
                'specify one label to use for all images or a list of labels of the ' + \
                'same length as number of images. Received {} labels '.format(len(label)) + \
                'but {} images are to be generated.'.format(num_images)
            if len(label) == 1:
                latent_labels.fill_(label[0])
            else:
                latent_labels = torch.tensor(label).to(latent_labels)
        self.Gs.set_truncation(
            truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        with torch.no_grad():
            generated = self.Gs(latents=latents, labels=latent_labels)
        assert generated.dim() - 2 == 2, 'Can only generate images when using a ' + \
            'network built for 2-dimensional data.'
        assert generated.dim() == 4, 'Only generators that produce 2d data ' + \
            'can be used to generate images.'
        return utils.tensor_to_PIL(generated, pixel_min=pixel_min, pixel_max=pixel_max)

    def add_tensorboard_image_logging(self, 
                                      name, 
                                      interval,
                                      num_images,
                                      resize=256,
                                      seed=None,
                                      truncation_psi=None,
                                      truncation_cutoff=None,
                                      label=None,
                                      pixel_min=-1,
                                      pixel_max=1):
        def callback(seen):
            if seen % interval == 0:
                images = self.generate_images(
                    num_images=num_images,
                    seed=seed,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    label=label,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max
                )
                image = utils.stack_images_PIL(images, individual_img_size=resize)
                image = torchvision.transforms.ToTensor()(image)
                self.tb_writer.add_image(name, image, self.seen)
                image2 = utils.stack_images_PIL(images) 
                image2.save('./runs/%d.png'%(self.seen)) #将图片保存到指定文件夹
        self.callbacks.append(callback)

    def save_checkpoint(self, dir_path):
        """
        Save the current state of this trainer as a checkpoint.
        NOTE: The dataset can not be serialized and saved so this has to be reconstructed and given when loading this checkpoint.
        Arguments:
            dir_path (str): The checkpoint path.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            assert os.path.isdir(dir_path), '`dir_path` points to a file.'
        kwargs = self.kwargs.copy()
        # Update arguments that may have changed since construction
        kwargs.update(seen=self.seen,pl_avg=float(self.pl_avg))
        with open(os.path.join(dir_path, 'kwargs.json'), 'w') as fp:
            json.dump(kwargs, fp)
        torch.save(self.G_opt.state_dict(), os.path.join(dir_path, 'G_opt.pth'))
        torch.save(self.D_opt.state_dict(), os.path.join(dir_path, 'D_opt.pth'))
        models.save(self.G, os.path.join(dir_path, 'G.pth'))
        models.save(self.D, os.path.join(dir_path, 'D.pth'))
        if self.Gs is not None:
            models.save(self.Gs, os.path.join(dir_path, 'Gs.pth'))

    @classmethod
    def load_checkpoint(cls, checkpoint_path, dataset, **kwargs):
        """
        Load a checkpoint into a new Trainer object and return that
        object. If the path specified points at a folder containing
        multiple checkpoints, the latest one will be used.
        The dataset can not be serialized and saved so it is required
        to be explicitly given when loading a checkpoint.
        Arguments:
            checkpoint_path (str): Path to a checkpoint or to a folder
                containing one or more checkpoints.
            dataset (indexable): The dataset to use.
            **kwargs (keyword arguments): Any other arguments to override
                the ones saved in the checkpoint. Useful for when training
                is continued on a different device or when distributed training
                is changed.
        """
        checkpoint_path = _find_checkpoint(checkpoint_path)
        _is_checkpoint(checkpoint_path, enforce=True)
        with open(os.path.join(checkpoint_path, 'kwargs.json'), 'r') as fp:
            loaded_kwargs = json.load(fp)
        loaded_kwargs.update(**kwargs)
        device = torch.device('cpu')
        if loaded_kwargs['device']:
            device = torch.device(loaded_kwargs['device'][0])
        for name in ['G', 'D']:
            fpath = os.path.join(checkpoint_path, name + '.pth')
            loaded_kwargs[name] = models.load(fpath, map_location=device)
        if os.path.exists(os.path.join(checkpoint_path, 'Gs.pth')):
            loaded_kwargs['Gs'] = models.load(
                os.path.join(checkpoint_path, 'Gs.pth'),
                map_location=device if loaded_kwargs['Gs_device'] is None \
                    else torch.device(loaded_kwargs['Gs_device'])
            )
        obj = cls(dataset=dataset, **loaded_kwargs)
        for name in ['G_opt', 'D_opt']:
            fpath = os.path.join(checkpoint_path, name + '.pth')
            state_dict = torch.load(fpath, map_location=device)
            getattr(obj, name).load_state_dict(state_dict)
        return obj

#----------------------------------------------------------------------------
# Checkpoint helper functions


def _is_checkpoint(dir_path, enforce=False):
    if not dir_path:
        if enforce:
            raise ValueError('Not a checkpoint.')
        return False
    if not os.path.exists(dir_path):
        if enforce:
            raise FileNotFoundError('{} could not be found.'.format(dir_path))
        return False
    if not os.path.isdir(dir_path):
        if enforce:
            raise NotADirectoryError('{} is not a directory.'.format(dir_path))
        return False
    fnames = os.listdir(dir_path)
    for fname in ['G.pth', 'D.pth', 'G_opt.pth', 'D_opt.pth', 'kwargs.json']:
        if fname not in fnames:
            if enforce:
                raise FileNotFoundError(
                    'Could not find {} in {}.'.format(fname, dir_path))
            return False
    return True


def _find_checkpoint(dir_path):
    if not dir_path:
        return None
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return None
    if _is_checkpoint(dir_path):
        return dir_path
    checkpoint_names = []
    for name in os.listdir(dir_path):
        if _is_checkpoint(os.path.join(dir_path, name)):
            checkpoint_names.append(name)
    if not checkpoint_names:
        return None
    def get_iteration(name):
        return int(name.split('_')[0])
    def get_timestamp(name):
        return '_'.join(name.split('_')[1:])
    # Python sort is stable, meaning that this sort operation will guarantee that the order of values after the first
    # sort will stay for a set of values that have the same key value.
    checkpoint_names = sorted(sorted(checkpoint_names, key=get_iteration), key=get_timestamp) #按照迭代次数或时间排序
    x = os.path.join(dir_path, checkpoint_names[-1])
    print(x)
    return x

#----------------------------------------------------------------------------
# Reg and loss function fetchers

def build_opt(net, opt_class, opt_kwargs, reg, reg_interval):
    opt_kwargs['lr'] = opt_kwargs.get('lr', 1e-3)
    if reg not in [None, False] and reg_interval:
        mb_ratio = reg_interval / (reg_interval + 1.)
        opt_kwargs['lr'] *= mb_ratio
        if 'momentum' in opt_kwargs:
            opt_kwargs['momentum'] = opt_kwargs['momentum'] ** mb_ratio
        if 'betas' in opt_kwargs:
            betas = opt_kwargs['betas']
            opt_kwargs['betas'] = (betas[0] ** mb_ratio, betas[1] ** mb_ratio)
    if isinstance(opt_class, str):
        opt_class = getattr(torch.optim, opt_class.title())
    return opt_class(net.parameters(), **opt_kwargs) # 直接通过访问属性获得对应文件夹下的类,通过__init__.py导入

#----------------------------------------------------------------------------
# Reg and loss function fetchers

_LOSS_FNS = {
    'G': {
        'logistic': loss_fns.G_logistic,
        'logistic_ns': loss_fns.G_logistic_ns,
        'wgan': loss_fns.G_wgan
    },
    'D': {
        'logistic': loss_fns.D_logistic,
        'wgan': loss_fns.D_wgan
    }
}

def get_loss_fn(net, loss):
    if callable(loss):
        return loss
    net = net.upper()
    assert net in ['G', 'D'], 'Unknown net type {}'.format(net)
    loss = loss.lower()
    for name in _LOSS_FNS[net].keys():
        if loss == name:
            return _LOSS_FNS[net][name]
    raise ValueError('Unknow {} loss {}'.format(net, loss))

_REG_FNS = {
    'G': {'pathreg': loss_fns.G_pathreg},
    'D': { 'r1': loss_fns.D_r1, 'r2': loss_fns.D_r2, 'gp': loss_fns.D_gp,}
    }

def get_reg_fn(net, reg, **kwargs):
    if reg is None:
        return None
    if callable(reg):
        functools.partial(reg, **kwargs)
    net = net.upper()
    assert net in ['G', 'D'], 'Unknown net type {}'.format(net)
    reg = reg.lower()
    gamma = None
    for name in _REG_FNS[net].keys():
        if reg.startswith(name):
            gamma_chars = [c for c in reg.replace(name, '') if c.isdigit() or c == '.']
            if gamma_chars:
                kwargs.update(gamma=float(''.join(gamma_chars)))
            return functools.partial(_REG_FNS[net][name], **kwargs)
    raise ValueError('Unknow regularizer {}'.format(reg))
