import torch
import tqdm

from torch.utils.tensorboard import SummaryWriter

from modules.optimization import build_optimizer, build_scheduler
from .common_utils import load_data_to_gpu
from .evaluator import Evaluator


class Trainer:
    def __init__(self, model, train_loader, val_loader, logger, args, optim_cfg):
        self.model = model.cuda()
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.logger = logger
        self.val_freq = args.val_freq
        self.args = args
        self.optim_cfg = optim_cfg  

        if self.optim_cfg.LR_SCHEDULER.NAME == 'OneCycleLR':
            num_steps_per_epoch = len(self.train_dataloader)
            optim_cfg.LR_SCHEDULER.PARAMS['total_steps'] = num_steps_per_epoch * optim_cfg.NUM_EPOCHS
        self.optimizer = build_optimizer(self.model, optim_cfg.OPTIMIZER)
        self.lr_scheduler = build_scheduler(self.optimizer, optim_cfg.LR_SCHEDULER)

        self.start_epoch = 1
        self.accumulated_iter = 0
        if args.ckpt_path is not None:
            self.start_epoch, self.accumulated_iter = \
                self.model.load_ckpt_with_optimizer(args.ckpt_path, self.optimizer, self.lr_scheduler)
        
        self.ckpt_save_interval = args.ckpt_save_interval
        self.ckpt_dir = args.output_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        tb_dir = args.output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dict = SummaryWriter(log_dir=tb_dir)

        self.evaluator = Evaluator(model=model,
                                   dataloader=val_loader,
                                   output_path=args.output_dir,
                                   logger=self.logger)


    def train_one_epoch(self, tbar):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm.tqdm(total=len(self.train_dataloader), desc="Training", dynamic_ncols=True)
        disp_dict = {}
        
        for i, batch_dict in enumerate(self.train_dataloader):
            batch_dict = load_data_to_gpu(batch_dict)

            self.optimizer.zero_grad()
            loss_dict = self.model(batch_dict)
            loss = loss_dict['loss']
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
            self.accumulated_iter += i
            self.tb_dict.add_scalar('Loss/train', loss.item(), self.accumulated_iter)
            
            pbar.set_postfix({loss_key: loss_val.item() for loss_key, loss_val in loss_dict.items()})
            pbar.update(1)

        return running_loss / len(self.train_dataloader)


    def save_ckpt(self, epoch):
        checkpoint_file = self.ckpt_dir / "epoch_{}.pth".format(epoch)
        self.model.save_ckpt(checkpoint_file, self.optimizer, self.lr_scheduler, epoch, self.accumulated_iter)

        saved_ckpt_list = list(self.ckpt_dir.glob("*.pth"))
        if len(saved_ckpt_list) > self.args.max_ckpt_save_num:
            saved_ckpt_list = sorted(saved_ckpt_list, key=lambda x: x.stat().st_ctime)
            for ckpt_file in saved_ckpt_list[:-self.args.max_ckpt_save_num]:
                ckpt_file.unlink()


    def train(self, val_while_training=True):
        total_epochs = self.optim_cfg.NUM_EPOCHS + 1
        
        with tqdm.trange(self.start_epoch, total_epochs, desc="epochs", dynamic_ncols=True) as tbar:
            for epoch in tbar:
                train_loss = self.train_one_epoch(tbar)
                self.lr_scheduler.step()

                self.logger.add_log('epoch {}/{} train_loss: {}'.format(epoch, total_epochs, train_loss), print_log=False)

                if val_while_training and epoch % self.val_freq == 0:
                    is_best = self.evaluator.eval_one_epoch(self.tb_dict, epoch)
                    if is_best:
                        self.model.save_ckpt(path=self.ckpt_dir / "best.pth")
                    
                if self.ckpt_save_interval > 0 and epoch % self.ckpt_save_interval == 0:
                    self.save_ckpt(epoch)

                self.logger.save_logs()
        
        if not val_while_training:
            self.evaluator.eval_one_epoch(self.tb_dict, epoch)
            self.save_ckpt(epoch)
        self.evaluator.close_eval(self.tb_dict)
