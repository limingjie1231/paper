
from argparse import ArgumentParser
import yaml
from easydict import EasyDict
import os
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from dataloader.pc_dataset import get_pc_model_class
from dataloader.dataset import get_model_class, get_collate_class
import numpy as np
import datetime
import torch
import importlib
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import warnings

warnings.filterwarnings("ignore")

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config
def build_loader(config):
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    train_config = config['dataset_params']['train_data_loader']
    val_config = config['dataset_params']['val_data_loader']
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None

    train_pt_dataset = pc_dataset(config, data_path=train_config['data_path'], imageset='train')
    val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val')
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset_type(train_pt_dataset, config, train_config),
        batch_size=train_config["batch_size"],
        collate_fn=get_collate_class(config['dataset_params']['collate_type']),
        shuffle=train_config["shuffle"],
        num_workers=train_config["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    # config['dataset_params']['training_size'] = len(train_dataset_loader) * len(configs.gpu)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=1),
        batch_size=val_config["batch_size"],
        collate_fn=get_collate_class(config['dataset_params']['collate_type']),
        shuffle=val_config["shuffle"],
        pin_memory=True,
        num_workers=val_config["num_workers"]
    )
    return train_dataset_loader, val_dataset_loader, test_dataset_loader
def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument('--log_dir', type=str, default='default', help='log location')
    parser.add_argument('--monitor', type=str, default='val/mIoU', help='the maximum metric')
    parser.add_argument('--save_top_k', type=int, default=3,help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='fine tune mode')
    parser.add_argument('--pretrain2d', action='store_true', default=False, help='use pre-trained 2d network')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--num_vote', type=int, default=1)
    parser.add_argument('--stop_patience', type=int, default=50, help='patience for stop training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args


    return EasyDict(config)


if __name__ == '__main__':
    configs = parse_config()
    print(configs)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))

    log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.log_dir, default_hp_metric=False)
    os.makedirs(f'{log_folder}/{configs.log_dir}', exist_ok=True)
    profiler = SimpleProfiler(f'{log_folder}/{configs.log_dir}/profiler.txt')
    np.set_printoptions(precision=4, suppress=True)

    backup_dir = os.path.join(log_folder, configs.log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(configs)
    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.Model(configs)
    num_gpu = len(configs.gpu)
    checkpoint_callback = ModelCheckpoint(
        monitor=configs.monitor,
        mode='max',
        save_last=True,
        save_top_k=configs.save_top_k)

    if configs.checkpoint is not None:
        print('load pre-trained model...')
        if configs.fine_tune or configs.test or configs.pretrain2d:
            my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs,
                                                     strict=(not configs.pretrain2d))
        else:
            # continue last training
            my_model = my_model.load_from_checkpoint(configs.checkpoint)
    if not configs.test:
        # init trainer
        print('Start training...')
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],
                             accelerator='gpu',
                             max_epochs=configs['train_params']['max_num_epochs'],
                             resume_from_checkpoint=configs.checkpoint if not configs.fine_tune and not configs.pretrain2d else None,
                             callbacks=[checkpoint_callback,
                                        LearningRateMonitor(logging_interval='step'),
                                        EarlyStopping(monitor=configs.monitor,
                                                      patience=configs.stop_patience,
                                                      mode='max',
                                                      verbose=True),
                                        ] ,
                             logger=tb_logger,
                             profiler=profiler,
                             check_val_every_n_epoch=configs.check_val_every_n_epoch,
                             gradient_clip_val=1,
                             accumulate_grad_batches=1
                             )
        trainer.fit(my_model, train_dataset_loader, val_dataset_loader)

    else:
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],
                             accelerator='gpu',
                             resume_from_checkpoint=configs.checkpoint,
                             logger=tb_logger,
                             profiler=profiler)
        trainer.test(my_model, test_dataset_loader if configs.submit_to_server else val_dataset_loader)