# main.py  —— 二分类训练/测试版本
from __future__ import annotations
import argparse, json, datetime, inspect, os
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# ==== 你的工程内模块（保持原路径）====
import models
import data_loader
import logger
import utils

# -------------------- CLI --------------------
parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type=str, choices=['train', 'test'], required=True, help='train or test')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)

# evaluation / files
parser.add_argument('--weight_file', type=str, help='path of model weight')
parser.add_argument('--result_file', type=str, default='./result.txt', help='file to save predictions')
parser.add_argument('--log_file', type=str, default='exp.log', help='path of log file')

# model-related (保持你的参数接口)
parser.add_argument('--kernel_size', type=int)
parser.add_argument('--pooling_method', type=str, default='attention')  # attention/mean
parser.add_argument('--alpha', type=float)

# ==== 新增：目标类型与类别数 ====
parser.add_argument('--objective', type=str, choices=['classification', 'regression'], default='classification')  ##### [CHANGE]
parser.add_argument('--num_classes', type=int, default=2)  ##### [CHANGE]
parser.add_argument('--pos_class', type=int, default=1, help='positive class id for metrics')  ##### [CHANGE]

args = parser.parse_args()
config = json.load(open('./config.json', 'r'))

# -------------------- 小工具 --------------------
def to_device(batch_dict: Dict[str, Any], device):
    """把 dict 里的张量递归搬到 device。"""
    out = {}
    for k, v in batch_dict.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def get_kwargs(model_class):
    """只保留模型 __init__ 里接受的参数"""
    model_args = list(inspect.signature(model_class.__init__).parameters.keys())
    shell_args = dict(args._get_kwargs())
    kwargs = {k: v for k, v in shell_args.items() if k in model_args}
    # 也可把 config 里的模型超参合并进来（可选）
    for k, v in config.get('model_kwargs', {}).items():
        if k in model_args:
            kwargs[k] = v
    return kwargs

# -------------------- 指标计算（分类） --------------------
def classification_metrics(all_labels, all_probs, pos_class=1):
    """
    all_labels: (N,) int
    all_probs:  (N,) 概率（正类）
    """
    preds = (all_probs >= 0.5).astype(np.int64)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, pos_label=pos_class)
    # AUC：需要同时存在两类
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) == 2 else 0.0
    return acc, f1, auc

# -------------------- 写结果（兼容分类） --------------------
def write_result(fs, pred_dict, attr):
    """
    兼容：pred_dict 至少包含
      - 'label': LongTensor [B]
      - 'prob':  FloatTensor [B]  (正类概率)
      - 'pred':  LongTensor [B]   (预测类别)
    """
    label = pred_dict['label'].detach().cpu().numpy().reshape(-1)
    prob = pred_dict['prob'].detach().cpu().numpy().reshape(-1)
    pred = pred_dict['pred'].detach().cpu().numpy().reshape(-1)
    for i in range(label.shape[0]):
        # 保存：真实标签  预测正类概率  预测类别
        fs.write(f"{int(label[i])} {prob[i]:.6f} {int(pred[i])}\n")

# -------------------- 训练/评估 --------------------
def train(model, elogger, train_set, eval_set):
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        print(f"Training on epoch {epoch}")
        model.train()
        epoch_loss = 0.0
        seen_batches = 0

        for input_file in train_set:
            print(f"Train on file {input_file}")
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0
            for idx, (attr, traj) in enumerate(data_iter):
                attr = to_device(attr, device)
                traj = to_device(traj, device)

                # 期望 model.eval_on_batch 返回 (pred_dict, loss)
                # 其中 pred_dict['label'] 为 LongTensor [B]
                #      pred_dict['prob']  为 FloatTensor [B] (正类概率)
                #      pred_dict['pred']  为 LongTensor [B] (预测类别)
                pred_dict, loss = model.eval_on_batch(attr, traj, config)  ##### [CHANGE] 确保内部按交叉熵计算

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                running_loss += loss_val
                epoch_loss += loss_val
                seen_batches += 1

                print('\r Progress {:.2f}%, average loss {:.6f}'.format(
                    (idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0)
                ), end='')

            print()
            elogger.log('Training Epoch {}, File {}, Loss {:.6f}'.format(
                epoch, input_file, running_loss / (idx + 1.0)
            ))

        # 每个 epoch 做一次验证
        eval_logs = evaluate(model, elogger, eval_set, save_result=False)
        elogger.log('[Eval] Epoch {} | loss {:.6f} | acc {:.4f} | f1 {:.4f} | auc {:.4f}'.format(
            epoch, eval_logs['loss'], eval_logs['acc'], eval_logs['f1'], eval_logs['auc']
        ))

        # 每个 epoch 保存一次权重
        weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
        elogger.log('Save weight file {}'.format(weight_name))
        os.makedirs('./saved_weights', exist_ok=True)
        torch.save(model.state_dict(), './saved_weights/' + weight_name)

def evaluate(model, elogger, files, save_result=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if save_result:
        fs = open('%s' % args.result_file, 'w')

    all_labels, all_probs = [], []
    total_loss, total_batches = 0.0, 0

    with torch.no_grad():
        for input_file in files:
            running_loss = 0.0
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            for idx, (attr, traj) in enumerate(data_iter):
                attr = to_device(attr, device)
                traj = to_device(traj, device)

                pred_dict, loss = model.eval_on_batch(attr, traj, config)  ##### [CHANGE] 二分类路径
                running_loss += loss.item()
                total_loss += loss.item()
                total_batches += 1

                # 累积指标
                y = pred_dict['label'].detach().cpu().numpy().reshape(-1)
                p1 = pred_dict['prob'].detach().cpu().numpy().reshape(-1)
                all_labels.append(y)
                all_probs.append(p1)

                if save_result:
                    write_result(fs, pred_dict, attr)

            avg_file_loss = running_loss / (idx + 1.0)
            print('Evaluate on file {}, loss {:.6f}'.format(input_file, avg_file_loss))
            elogger.log('Evaluate File {}, Loss {:.6f}'.format(input_file, avg_file_loss))

    if save_result:
        fs.close()

    # 汇总指标
    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        acc, f1, auc = classification_metrics(all_labels, all_probs, pos_class=args.pos_class)
    else:
        acc = f1 = auc = 0.0

    logs = {
        'loss': (total_loss / max(total_batches, 1)),
        'acc': acc, 'f1': f1, 'auc': auc
    }
    print('[Eval Summary] loss {:.6f} | acc {:.4f} | f1 {:.4f} | auc {:.4f}'.format(
        logs['loss'], logs['acc'], logs['f1'], logs['auc']
    ))
    return logs

def run():
    # 生成模型
    kwargs = get_kwargs(models.DeepTTE.Net)
    # 强制告诉模型现在是分类任务与类别数（如果模型支持该入参）
    kwargs.update(dict(objective=args.objective, num_classes=args.num_classes))  ##### [CHANGE]

    # 模型对象
    model = models.DeepTTE.Net(**kwargs)
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        # 这里原本是已经分好的文件夹，我的是一个数组过来 dataset在哪里用？
        train(model, elogger, train_set=config['train_set'], eval_set=config['eval_set'])
    elif args.task == 'test':
        assert args.weight_file is not None, 'Please provide --weight_file for test.'
        model.load_state_dict(torch.load(args.weight_file, map_location='cpu'))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result=True)

if __name__ == '__main__':
    run()
