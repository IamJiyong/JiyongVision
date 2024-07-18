import torch
from torch.utils.tensorboard import SummaryWriter

from modules.dataset import load_data_to_gpu


class Trainer:
    def __init__(self, model, logger, args, cfgs):
        self.model = model.cuda()
        self.logger = logger
        self.val_freq = args.val_freq
        self.args = args
        self.cfgs = cfgs

        self.optimizer = build_optimizer(cfgs.OPTIMIZER, model)
        self.scheduler = build_scheduler(cfgs.LR_SCHEDULER, self.optimizer)

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train_one_epoch(self, dataloader, num_iter, tb_dict=SummaryWriter()):
        self.model.train()
        running_loss = 0.0
        for i, batch_dict in enumerate(dataloader):
            batch_dict = load_data_to_gpu(batch_dict)
            batch_dict = dataloader.dataset.mixup(batch_dict)

            self.optimizer.zero_grad()
            loss = self.model(batch_dict)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
            num_iter += i
            tb_dict.add_scalar('Loss/train', loss.item(), num_iter)

        return running_loss / len(dataloader), num_iter

    def train(self, train_dataloader, val_dataloader):
        tb_dict = SummaryWriter(log_dir=self.args.output_dir)
        best_val_acc_top1 = 0.0
        best_val_acc_top5 = 0.0
        best_epoch = 0
        num_iter = 0
        for epoch in range(0, self.cfgs.NUM_EPOCHS):
            train_loss, num_iter = self.train_one_epoch(train_dataloader, num_iter=num_iter, tb_dict=tb_dict)
            self.scheduler.step()

            self.logger.add_log('epoch {} train_loss: {}'.format(epoch, train_loss))

            if epoch % self.val_freq == 0:
                eval_dict = self.eval_one_epoch(val_dataloader)
                val_loss = eval_dict['loss']
                val_acc_top1 = eval_dict['top1_acc']
                val_acc_top5 = eval_dict['top5_acc']

                tb_dict.add_scalar('Loss/val', val_loss, epoch)
                tb_dict.add_scalar('Accuracy_top1/val', val_acc_top1, epoch)
                tb_dict.add_scalar('Accuracy_top5/val', val_acc_top5, epoch)
                self.logger.add_log('epoch {} val_loss: {}, val_acc_top1: {}, val_acc_top5: {}'.format(epoch, val_loss, val_acc_top1, val_acc_top5))

                if best_val_acc_top1 < val_acc_top1:
                    best_val_acc_top1 = val_acc_top1
                    best_val_acc_top5 = val_acc_top5
                    best_epoch = epoch
                    self.model.save_ckpt(self.args.output_dir / "best_model.pth")
                
                if self.args.save_every_ckpt:
                    self.model.save_ckpt(self.args.output_dir / "epoch_{}.pth".format(epoch))

            self.logger.save_logs()
        
        self.logger.add_log('Best acc at epoch {}'.format(best_epoch))
        self.logger.add_log('Best val acc top1: {}'.format(best_val_acc_top1))
        self.logger.add_log('Best val acc top5: {}'.format(best_val_acc_top5))
        self.logger.save_logs()

        tb_dict.close()

        return best_val_acc_top1, best_val_acc_top5
    
    def eval_one_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct_top5 = 0
        correct_top1 = 0
        total = 0
        with torch.no_grad():
            for i, batch_dict in enumerate(dataloader):
                batch_dict = load_data_to_gpu(batch_dict)

                if self.model.ensemble:
                    pred_dict, loss = self.model.forward_ensemble(batch_dict)
                else:
                    pred_dict, loss = self.model.forward(batch_dict)

                running_loss += loss.item()

                total += batch_dict['batch_size']
                top5_correct = torch.topk(pred_dict['pred_scores'], 5).indices
                correct_top5 += torch.sum(top5_correct == batch_dict['target'].unsqueeze(1)).item()
                top1_correct = torch.argmax(pred_dict['pred_scores'], dim=1)
                correct_top1 += torch.sum(top1_correct == batch_dict['target']).item()

        eval_dict = {
            'loss': running_loss / len(dataloader),
            'top1_acc': correct_top1 / total,
            'top5_acc': correct_top5 / total
        }
        return eval_dict


def build_optimizer(optimizer_config, model):
    return getattr(torch.optim, optimizer_config.NAME)(model.parameters(), **optimizer_config.PARAMS)

def build_scheduler(scheduler_config, optimizer):
    return getattr(torch.optim.lr_scheduler, scheduler_config.NAME)(optimizer, **scheduler_config.PARAMS)