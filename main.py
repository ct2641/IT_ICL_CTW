# main.py
import argparse

from train import train_model
from evaluate import evaluate_model
from data_gen import data_gen
import config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ICL of context trees")
    parser.add_argument("--mode", choices=['data_gen','train', 'evaluate'], default='train', help="Mode: data generation, train, or evaluate the model")
    parser.add_argument('--config', type=str, required=True, help='Name of the configuration class in config.py')    
    args = parser.parse_args()

    config_class = getattr(config, args.config)
    cfg = config_class()

    if args.mode == 'data_gen':
        data_gen(cfg)
    elif args.mode == 'train':   
        train_model(cfg)
    elif args.mode == 'evaluate':
        evaluate_model(cfg)   

# now should call main.py --config Config_test --mode data_gen 