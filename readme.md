# 文件结构(基础文件)

## (1). defaults.py && configs文件夹

- 放配置文件, 其中defaults.py用的是:
>from yacs.config import CfgNode as CN

用于保存常用配置

- configs内用的是yaml

用于保存单次实验配置

## (2). Manuscript.md

常用预训练模型的记录，转换自官方git，转换文件:run_convert_from_tf.py

## (3). 运行文件 (run文件夹/run_xxx.py)

需要拉出文件夹下才可使用

- run_convert_from_tf.py

完成模型转换

- run_gnerator.py

用于生成类似官方论文展示的混合生成图

- run_metrics.py

用于运行评测数据

- run_projector.py

用于映射图片到潜空间

- run_training.py

```
def get_dataset(): #读取图片型数据集,可以按不同分辨率读取

def get_models(args): #获得模型

def get_trainer(args): #获得训练器, 来自stylegan2.train.Trainer()
	# 做了一些训练可选设置, 计算各种指标

def run(args):
	#创建trainer对象，开始训练

```

## (4). 核心文件(stylegan2)

- train.py
  1. 把G,GS(就是Gmap),D都封装到trainer类中
  2. 把数据也封装到trainer类中
  3. 产生潜变量的类(继承自utils.PriorGenerator, 可以输出label)

- models.py

  1. 基类（BaseModel）: 

  > 将必要的配置参数放进self.kwargs中(以字典的形式)，针对不同的模型更新参数kwargs

  2. Generator类: 包含Gm和Gs

- modules.py

  > 重写了大部分layer的基本操作，666

  1. 用一个函数实现equalize learning rate: get_weight_and_coef()
  2. 


- loss_fns.py

  non-saturating logistic:

  > F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))

  判断它是真的

  saturating logistic:	

  >  \- F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))

  判断它是假的，取反

## (5).Manuscript

训练时命令行参数设置

### Train

python run_training.py 

> --resume will look in the checkpoint directory if we specified one and attempt to load the latest checkpoint before continuing to train. 

### Generating Images

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

### Project 

python run_projector.py project_generated_images --network=Gs.pth --seeds=0,1,5
>generated images

python run_projector.py project_real_images --network=Gs.pth --data-dir=path/to/image_folder
>real images



