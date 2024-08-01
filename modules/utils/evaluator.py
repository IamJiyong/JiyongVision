import torch

from tqdm import tqdm

from .common_utils import load_data_to_gpu

class Evaluator(object):
    def __init__(self, model, dataloader, output_path, logger):
        self.model = model.cuda()
        self.dataloader = dataloader
        self.logger = logger

        self.class_names = self.dataloader.dataset.class_names

        self.task = model.task
        if self.task == 'classification':
            self.best_eval_key = 'top1_acc'
            self.higher_better = True
        elif self.task == 'detection':
            self.best_eval_key = 'mAP'
            self.higher_better = True
        else:
            raise ValueError("Invalid task")
        
        self.results_path = output_path / 'eval'
        self.results_path.mkdir(exist_ok=True)
        
        self.best_eval_value = 0.0
        self.best_eval_dict = {}
        self.best_epoch = 0
    

    def log_result(self, eval_dict):
        log_message = ''
        for key, val in eval_dict.items():
            log_message += f'{key}: {val}\n'
        self.logger.add_log(log_message[:-2])
        self.logger.save_logs()
    
    
    def eval_one_epoch(self, tb_dict, epoch):
        if self.task == 'classification':
            eval_dict = self.eval_classification()
        elif self.task == 'detection':
            eval_dict = self.eval_detection(key=epoch)
            
        assert self.best_eval_key in eval_dict, "best_eval_key not in eval_dict"
        
        eval_value = eval_dict[self.best_eval_key]
        is_best = self.best_eval_value < eval_value if self.higher_better\
            else self.best_eval_value > eval_value
        
        if is_best:
            self.best_eval_value = eval_value
            self.best_eval_dict = eval_dict
            self.best_epoch = epoch
        
        for key, value in eval_dict.items():
            tb_dict.add_scalar(f'{key}/val', value, epoch)
        
        self.logger.add_log(f'epoch {epoch}: ')
        self.log_result(eval_dict)

        return is_best


    def eval_classification(self, key=None):
        self.model.eval()
        
        # TODO: save results to file
        # results_path = self.results_path / key / 'results'
        # results_path.mkdir(exist_ok=True)
        
        running_loss = 0.0
        correct_top5 = 0
        correct_top1 = 0
        total = 0
        
        for batch_dict in tqdm(self.dataloader, desc="Detection Evaluation"):
            batch_dict = load_data_to_gpu(batch_dict)

            with torch.no_grad():
                pred_dict, loss_dict = self.model.forward(batch_dict)
            
            loss = loss_dict['loss']
            running_loss += loss.item()

            total += batch_dict['batch_size']
            top5_correct = torch.topk(pred_dict['pred_scores'], 5).indices
            correct_top5 += torch.sum(top5_correct == batch_dict['target'].unsqueeze(1)).item()
            top1_correct = torch.argmax(pred_dict['pred_scores'], dim=1)
            correct_top1 += torch.sum(top1_correct == batch_dict['target']).item()

        eval_dict = {
            'loss': running_loss / len(self.dataloader),
            'top1_acc': correct_top1 / total,
            'top5_acc': correct_top5 / total
        }
        return eval_dict
    

    def _get_result_files(self, results_dir=None):
        if results_dir is not None:
            results_files = [results_dir / f'{class_name}.txt' for class_name in self.class_names]
        else:
            results_files = [self.results_path / f'{class_name}.txt' for class_name in self.class_names]
        return results_files


    def eval_detection(self, key=None):
        self.model.eval()

        result_files = self._get_result_files()
        opened_results_files = [open(f, mode='w') for f in result_files]

        for batch_dict in tqdm(self.dataloader, desc="Detection Evaluation"):
            batch_dict = load_data_to_gpu(batch_dict)
            
            with torch.no_grad():
                pred_dicts, _ = self.model.forward(batch_dict)

            for batch_idx, pd in enumerate(pred_dicts):
                pred_boxes = pd['pred_boxes']
                pred_scores = pd['pred_scores']
                pred_labels = pd['pred_labels']

                img_id = batch_dict['img_id'][batch_idx][1]
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    print(img_id, "{:.3f}".format(score), "{:.1f} {:.1f} {:.1f} {:.1f}".format(*box),\
                          file=opened_results_files[label], sep=' ')
                
                if False: # visualize results
                    import cv2
                    import numpy as np
                    from matplotlib import pyplot as plt
                    # load image
                    original_img = cv2.imread(str(self.dataloader.dataset._imgpath) % batch_dict['img_id'][batch_idx], cv2.IMREAD_COLOR)
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    # draw boxes
                    pallet = np.random.randint(0, 255, (len(self.class_names), 3))
                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        box = [int(b) for b in box]
                        cv2.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), pallet[label].tolist(), 2)
                        cv2.putText(original_img, self.class_names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pallet[label].tolist(), 2)
                        cv2.putText(original_img, "{:.3f}".format(score), (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pallet[label].tolist(), 2)
                    # save image
                    cv2.imwrite(f'visualize.jpg', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))        
    
        for f in opened_results_files:
            f.close()
        
        eval_dict = self.calculate_ap()
        return eval_dict


    def calculate_ap(self, results_dir=None):
        print("Calculating AP...")
        result_files = self._get_result_files(results_dir)

        eval_dict = {}
        for cls_idx, file_path in enumerate(result_files):
            class_name = self.class_names[cls_idx]
            rec, prec, ap = self.dataloader.dataset.evaluate(file_path, class_name)
            eval_dict[class_name] = ap
        eval_dict['mAP'] = sum(eval_dict.values()) / len(eval_dict)
        return eval_dict
    

    def close_eval(self, tb_dict):
        self.logger.add_log(f'Best {self.best_eval_key} at epoch {self.best_epoch}')
        for key, value in self.best_eval_dict.items():
            self.logger.add_log(f'Best {key}: {value}')
        self.logger.save_logs()
        tb_dict.close()
