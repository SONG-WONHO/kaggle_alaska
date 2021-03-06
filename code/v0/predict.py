import os, sys, argparse, json
from pprint import pprint
import warnings

import torch.nn as nn

from data import *
from transform import get_transform
from model import get_model
from learner import Learner
from utils import *

warnings.filterwarnings("ignore")


class CFG:
    # path
    root_path = "./input/"
    save_path = './submission/'
    sub_name = 'submission.csv'

    # learning
    batch_size = 64
    workers = 0
    seed = 42

    typ = 'metric'


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--root-path', default=CFG.root_path,
                        help="root path")
    parser.add_argument('--save-path', default=CFG.save_path,
                        help="save path")
    parser.add_argument('--sub-name', default=CFG.sub_name,
                        help="submission name")

    # learning
    parser.add_argument('--batch-size', default=CFG.batch_size, type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument("--workers", default=CFG.workers, type=int,
                        help=f"number of workers({CFG.workers})")
    parser.add_argument("--seed", default=CFG.seed, type=int,
                        help=f"seed({CFG.seed})")

    # version
    parser.add_argument('--version', type=int)
    parser.add_argument('--exp-id', type=int)

    # typ
    parser.add_argument('--typ', type=str, default=CFG.typ, choices=['metric', 'loss'])

    args = parser.parse_args()

    CFG.root_path = args.root_path
    CFG.save_path = args.save_path
    CFG.sub_name = args.sub_name

    CFG.batch_size = args.batch_size
    CFG.workers = args.workers
    CFG.seed = args.seed

    CFG.model_path = f"./model/v{args.version}/exp_{args.exp_id}/"
    CFG.log_path = f"./log/v{args.version}/exp_{args.exp_id}/"

    CFG.typ = args.typ

    # get device
    CFG.device = get_device()

    # load train environment
    env = json.load(open(os.path.join(CFG.log_path, 'CFG.json'), 'r'))
    for k, v in env.items(): setattr(CFG, k, v)

    score = pd.read_csv(os.path.join(CFG.log_path, "log.csv")).sort_values(f'val_{CFG.typ}', ascending=False).iloc[0]
    CFG.sub_name = f"submission.ver_{args.version}.exp_{args.exp_id}.loss_{score['val_loss']:.4f}.metric_{score['val_metric']:.4f}.csv"

    pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})

    ### seed all
    seed_everything(CFG.seed)

    ### Data related logic
    # load data
    print("Load Raw Data")
    _, _, test_df = load_data(CFG, CFG.train_sample_size, CFG.valid_sample_size)

    # get transform
    print("Get Transform")
    _, test_transforms = get_transform(CFG)

    # dataset
    print("Get Dataset")
    tst_data = Alaska2Dataset(CFG, test_df, test_transforms)

    ### learner
    model_name = 'model.best.pt'
    learner = Learner(CFG)
    learner.load(os.path.join(CFG.model_path, model_name), f"model_state_dict")

    ### predicton
    ss_df = pd.read_csv(os.path.join(CFG.root_path, "sample_submission.csv"))

    test_preds = learner.predict(tst_data)

    # multi classification
    test_preds_multi = nn.Softmax()(torch.tensor(test_preds[:, :-1]))[:, 1:].sum(-1).numpy()

    ss_df['Label'] = test_preds_multi
    ss_df.to_csv(os.path.join(CFG.save_path, CFG.sub_name), index=False)


if __name__ == '__main__':
    main()
