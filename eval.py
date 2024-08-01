from tensorboardX import SummaryWriter
import argparse

from datetime import datetime
from pathlib import Path

from modules.dataset import build_dataloader
from modules.model import build_network
from modules.utils.evaluator import Evaluator
from modules.utils.config import cfg_from_yaml_file, cfg
from modules.utils.logger import Logger


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser") 
    parser.add_argument("--cfg_file", type=str, help="Path to the config file")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to the results directory")
    parser.add_argument("--extra_tag", type=str, default="default", help="Extra tag for the current run")
    
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    
    parser.add_argument("--eval_all", action='store_true', help="Evaluate all checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Path to the checkpoint directory")

    args = parser.parse_args()
    cfgs = cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfgs


def main():
    args, cfgs = parse_config()

    if args.ckpt_path is not None:
        ckpt_name = args.ckpt_path.split('/')[-1].split('.')[0]
    if args.results_dir is not None:
        args.results_dir = Path(args.results_dir)
    
    model_name = args.cfg_file.split('/')[-1].split('.')[0]
    output_path = Path(cfg.ROOT_DIR) / "outputs" / (cfg.DATA_CONFIG.DATASET + '_models') / model_name / args.extra_tag
    output_path.mkdir(parents=True, exist_ok=True)

    if args.batch_size is None:
        args.batch_size = cfgs.OPTIMIZATION.BATCH_SIZE
    else:
        cfgs.OPTIMIZATION.BATCH_SIZE = args.batch_size
        
    logger_path = output_path / "{}_logs.txt".format(datetime.now().strftime("%Y%m%d%H%M%S"))
    logger = Logger(logger_path)
    logger.add_config(cfgs)

    val_dataloader = build_dataloader(root_dir=cfg.ROOT_DIR,
                                      data_config=cfgs.DATA_CONFIG,
                                      args=args,
                                      mode='val')
    
    model = build_network(cfgs.MODEL)

    if args.eval_all:
        if args.ckpt_dir is None:
            ckpt_dir = output_path / 'ckpt'
        else:
            ckpt_dir = Path(args.ckpt_dir)
            
        for ckpt_path in ckpt_dir.glob('*.pth'):
            ckpt_name = ckpt_path.stem
            logger.add_log(f'Evaluate checkpoint: {ckpt_name}')
            eval_single_ckpt(args, ckpt_name, output_path, logger, val_dataloader, model)
    else:
        eval_single_ckpt(args, ckpt_name, output_path, logger, val_dataloader, model)


def eval_single_ckpt(args, ckpt_name, output_path, logger, val_dataloader, model):
    if args.ckpt_path is not None:
        model.load_ckpt(args.ckpt_path)
        
    evaluator = Evaluator(model=model,
                          dataloader=val_dataloader,
                          output_path=output_path,
                          logger=logger)
    
    if evaluator.task == 'classification':
        assert args.ckpt_path is not None, "ckpt_path should be provided for classification task"
        eval_dict = evaluator.eval_classification(key=ckpt_name)

    elif evaluator.task == 'detection':
        if args.results_dir is not None:
            eval_dict = evaluator.calculate_ap(results_dir=args.results_dir)
        else:
            assert args.ckpt_path is not None, "ckpt_path or results_dir should be provided for detection task"
            eval_dict = evaluator.eval_detection(key=ckpt_name)

    else:
        raise ValueError("Invalid task")

    evaluator.log_result(eval_dict)


if __name__ == "__main__":
    main()
