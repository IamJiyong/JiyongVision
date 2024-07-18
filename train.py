import argparse
from datetime import datetime
from pathlib import Path

from modules.dataset import build_dataloader
from modules.model import build_network
from modules.utils.trainer import Trainer
from modules.utils.config import cfg_from_yaml_file, cfg
from modules.utils.logger import Logger


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser") 
    parser.add_argument("--cfg_file", type=str, help="Path to the config file")
    parser.add_argument("--extra_tag", type=str, default="default", help="Extra tag for the current run")
    parser.add_argument("--val_freq", type=int, default=5, help="Validation frequency")
    parser.add_argument("--save_every_ckpt", action="store_true", help="Save only the best model")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")

    args = parser.parse_args()
    cfgs = cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfgs


def main():
    args, cfgs = parse_config()

    output_path = Path(cfg.ROOT_DIR) / "outputs" / args.extra_tag
    output_path.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_path

    logger_path = output_path / "{}_logs.txt".format(datetime.now().strftime("%Y%m%d%H%M%S"))
    logger = Logger(logger_path)
    logger.add_config(cfgs)

    if args.batch_size is None:
        args.batch_size = cfgs.OPTIMIZATION.BATCH_SIZE
    
    train_dataloader = build_dataloader(root_dir=cfg.ROOT_DIR,
                                        data_config=cfgs.DATA_CONFIG,
                                        args=args,
                                        mode='train')
    val_dataloader = build_dataloader(root_dir=cfg.ROOT_DIR,
                                      data_config=cfgs.DATA_CONFIG,
                                      args=args,
                                      mode='val')
    
    model = build_network(cfgs.MODEL)

    trainer = Trainer(model, logger, args, cfgs.OPTIMIZATION)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
