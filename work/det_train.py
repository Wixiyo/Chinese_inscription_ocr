from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import os.path as osp

def build_cfg():
    cfg = Config.fromfile('.././configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015-test.py')

    # 存放输出结果和日志的目录
    cfg.work_dir = '.././demo/det'

    # 初始学习率 0.001 是针对 8 个 GPU 训练的
    # 如果只有一个 GPU，则除以8
    cfg.optimizer.lr = 0.001 / 8
    cfg.lr_config.warmup = None

    # 每训练40张图像，记录一次日志
    cfg.log_config.interval = 40

    # 设置随机数种子
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    print(cfg.pretty_text)
    return cfg

def train(cfg):
    # 建立数据集
    datasets = [build_dataset(cfg.data.train)]

    # 建立模型
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # 创建新目录，保存训练结果
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':

    cfg = build_cfg()
    train(cfg)