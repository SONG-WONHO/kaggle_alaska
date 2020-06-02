"""Train Main
"""
import os
import sys
import json
import warnings
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim

from data import *
from transform import get_transform
from model import get_model
from learner import Learner
from utils import *

warnings.filterwarnings("ignore")


class CFG:
    # path
    root_path = "./input/"
    log_path = './log/'
    model_path = './model/'

    # preprocess
    image_size = 512

    # model
    model_name = "BaseModel"

    # train
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 40
    train_sample_size = 30
    valid_sample_size = 30

    # etc
    seed = 42
    workers = 1
    num_targets = 10


def main():
    """main function
    """

    ### header
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--root-path', default=CFG.root_path,
                        help="root path")
    parser.add_argument('--log-path', default=CFG.log_path,
                        help="log path")
    parser.add_argument('--model-path', default=CFG.model_path,
                        help="model path")
    parser.add_argument('--pretrained-path',
                        help='pretrained path')

    # image
    parser.add_argument('--transform-version', default=0, type=int,
                        help="image transform version ex) 0, 1, 2 ...")

    # learning
    parser.add_argument('--batch-size', default=CFG.batch_size, type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument('--learning-rate', default=CFG.learning_rate, type=float,
                        help=f"learning rate({CFG.learning_rate})")
    parser.add_argument('--num-epochs', default=CFG.num_epochs, type=int,
                        help=f"number of epochs({CFG.num_epochs})")
    parser.add_argument('--train-sample-size', default=CFG.train_sample_size, type=int,
                        help=f"train sample size ({CFG.train_sample_size})")
    parser.add_argument('--valid-sample-size', default=CFG.valid_sample_size, type=int,
                        help=f"valid sample size ({CFG.valid_sample_size})")

    # model

    # etc
    parser.add_argument("--workers", default=CFG.workers, type=int,
                        help=f"number of workers({CFG.workers})")
    parser.add_argument("--seed", default=CFG.seed, type=int,
                        help=f"seed({CFG.seed})")


    args = parser.parse_args()

    # path
    CFG.root_path = args.root_path
    CFG.log_path = args.log_path
    CFG.model_path = args.model_path
    CFG.pretrained_path = args.pretrained_path

    # image
    CFG.transform_version = args.transform_version

    # learning
    CFG.batch_size = args.batch_size
    CFG.learning_rate = args.learning_rate
    CFG.num_epochs = args.num_epochs
    CFG.train_sample_size = args.train_sample_size
    CFG.valid_sample_size = args.valid_sample_size

    # model

    # etc
    CFG.workers = args.workers
    CFG.seed = args.seed

    # get device
    CFG.device = get_device()

    # get version
    _, version, _ = sys.argv[0].split('/')
    CFG.version = version

    # update log path
    CFG.log_path = os.path.join(CFG.log_path, CFG.version)
    os.makedirs(CFG.log_path, exist_ok=True)
    CFG.log_path = os.path.join(CFG.log_path, f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
    os.makedirs(CFG.log_path, exist_ok=True)

    # update model path
    CFG.model_path = os.path.join(CFG.model_path, version)
    os.makedirs(CFG.model_path, exist_ok=True)
    CFG.model_path = os.path.join(CFG.model_path, f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
    os.makedirs(CFG.model_path, exist_ok=True)

    pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})
    json.dump(
        {k: v for k, v in dict(CFG.__dict__).items() if '__' not in k},
        open(os.path.join(CFG.log_path, 'CFG.json'), "w"))

    ### seed all
    seed_everything(CFG.seed)

    ### Data related logic
    # load data
    print("Load Raw Data")
    train_df, valid_df, test_df = load_data(CFG, CFG.train_sample_size, CFG.valid_sample_size)

    # get transform
    print("Get Transform")
    train_transforms, test_transforms = get_transform(CFG)

    # dataset
    print("Get Dataset")
    trn_data = Alaska2Dataset(CFG, train_df, train_transforms)
    val_data = Alaska2Dataset(CFG, valid_df, test_transforms)

    ### Model related logic
    # get learner
    learner = Learner(CFG)
    if CFG.pretrained_path:
        print("Load Pretrained Model")
        print(f"... Pretrained Info - {CFG.pretrained_path}")
        learner.load(CFG.pretrained_path, f"model_state_dict")

    # get model
    if CFG.pretrained_path:
        print(f"Get Model")
        model = learner.best_model
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        print(f"Get Model")
        model = get_model(CFG)

        net = [m for m in list(model.model.children()) if not isinstance(m, nn.BatchNorm2d)]
        modules = []

        for block in list(net[1].children()):
            modules.append(nn.Sequential(*[m for m in list(block.children()) if not isinstance(m, nn.BatchNorm2d)]))

        modules = nn.Sequential(*modules)
        net[1] = modules
        net = nn.Sequential(*net[:3])
        model.model = net
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(CFG.device)

    # get optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'BatchNorm.bias', 'BatchNorm.weight', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = optim.AdamW(optimizer_grouped_parameters, CFG.learning_rate)

    # get scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)

    ### Train related logic
    learner.train(trn_data, val_data, model, optimizer, scheduler)


if __name__ == "__main__":
    main()