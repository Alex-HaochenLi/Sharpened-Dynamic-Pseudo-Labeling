# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""
import argparse
import glob
import logging
import os
import re
import random


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from TweetNormalizer import normalizeTweet
from tqdm import trange
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from datasets import load_dataset
import sys
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, optimizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0.0
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    model.train()
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(args.device) for t in batch)
            tokens = batch[0]
            labels = batch[1]
            loss, _ = model(tokens, labels, sharpen=args.sharpen)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, checkpoint=str(global_step))
                        for key, value in results.items():
                            logger.info('loss before pseudo labeling %s', str(tr_loss - logging_loss))
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break

        logger.info('\n')
        logger.info('Loss of true label training %s', str(tr_loss / step))

        if args.semi and idx >= (args.pseudo_start - 1):
            # predict pseudo labels
            unlabel_data = [[torch.tensor(train_dataset.unlabel_examples[i].tweet_ids), torch.tensor(train_dataset.unlabel_examples[i].label)]
                            for i in range(len(train_dataset.unlabel_examples))]
            unlabel_sampler = SequentialSampler(unlabel_data)
            unlabel_dataloader = DataLoader(unlabel_data, sampler=unlabel_sampler, batch_size=args.train_batch_size * 2)
            hamming = []
            pos_count, neg_count = 0, 0
            std_logits = torch.zeros(11).to(args.device)
            for step, unlabel_data in enumerate(unlabel_dataloader):
                model.eval()
                tokens = unlabel_data[0].to(args.device)
                labels = unlabel_data[1].to(args.device)

                if args.naivepl:
                    with torch.no_grad():
                        _, logits = model(tokens, labels)
                    pseudo_labels_hamming = torch.sigmoid(logits).cpu().gt(args.threshold).int()
                    pos_pseudo = torch.sigmoid(logits).cpu().gt(args.threshold).int()
                    neg_pseudo = torch.sigmoid(logits).cpu().lt(1 - args.threshold).int() * 2
                    pseudo_labels = pos_pseudo + neg_pseudo
                elif not args.naivepl:
                    model.train()  # for calculating uncertainty
                    logits = []
                    with torch.no_grad():
                        for i in range(10):
                            _, logit = model(tokens, labels)
                            logits.append(logit)
                    logits = torch.stack(logits)
                    mean_logits = logits.mean(dim=0)
                    std_logits_tmp = torch.sigmoid(logits).std(dim=0)

                    if args.onlyups:
                        pseudo_labels_hamming = torch.sigmoid(mean_logits).gt(args.threshold).int()
                        pos_pseudo = torch.sigmoid(mean_logits).gt(args.threshold)
                        neg_pseudo = torch.sigmoid(mean_logits).lt(1 - args.threshold)

                        certain_pseudo = std_logits_tmp.lt(args.pos_certainty_threshold)
                        pos_pseudo = (pos_pseudo & certain_pseudo).int()
                        certain_pseudo = std_logits_tmp.lt(args.neg_certainty_threshold)
                        neg_pseudo = (neg_pseudo & certain_pseudo).int() * 2
                        pseudo_labels = pos_pseudo + neg_pseudo

                    else:
                        std_logits = step / (step + 1) * std_logits + 1 / (step + 1) * std_logits_tmp.mean(dim=0)

                        # pos_thres_weight = (std_logits.mean(dim=0) ** 0.8) / (std_logits.mean(dim=0) ** 0.8).max()
                        pos_thres_weight = (std_logits ** 0.5) / (std_logits ** 0.5).max()
                        pos_thres_weight = pos_thres_weight * args.threshold

                        # pseudo_labels_hamming = torch.sigmoid(mean_logits).cpu().gt(args.threshold).int()
                        pseudo_labels_hamming = torch.sigmoid(mean_logits).gt(torch.clamp(pos_thres_weight, min=1-args.threshold, max=0.95)).int()
                        pos_pseudo = torch.sigmoid(mean_logits).gt(torch.clamp(pos_thres_weight, min=1-args.threshold, max=0.95))
                        neg_pseudo = torch.sigmoid(mean_logits).lt(1 - args.threshold)

                        certain_pseudo = std_logits_tmp.lt(args.pos_certainty_threshold)
                        pos_pseudo = (pos_pseudo & certain_pseudo).int()
                        certain_pseudo = std_logits_tmp.lt(args.neg_certainty_threshold)
                        neg_pseudo = (neg_pseudo & certain_pseudo).int() * 2
                        pseudo_labels = pos_pseudo + neg_pseudo

                for i in range(len(pseudo_labels)):
                    if (pseudo_labels[i] == 1).any():
                        pos_count += (pseudo_labels[i] == 1).sum().item()
                    if (pseudo_labels[i] == 2).any():
                        neg_count += (pseudo_labels[i] == 2).sum().item()
                    if ((pseudo_labels[i] == 1) | (pseudo_labels[i] == 2)).any():
                        train_dataset.update_data(unlabel_data[0][i], pseudo_labels[i])
                        hamming.append(hamming_loss(pseudo_labels_hamming[i].cpu().numpy(), unlabel_data[1][i].cpu().numpy()))

            logger.info('POSITIVE pseudo labels: %s', str(pos_count))
            logger.info('NEGATIVE pseudo labels: %s', str(neg_count))
            if not args.onlyups and not args.naivepl:
                logger.info('positive threshold: %s', str(pos_thres_weight.cpu().numpy()))

            # train pseudo labels
            if len(train_dataset.pseudo_examples) > 0:
                logger.info('Hamming loss between pseudo labels and true labels: %s',
                            str(np.array(hamming).mean()))
                pseudo_loss = []
                pseudo_data = [[train_dataset.pseudo_examples[i].tweet_ids, train_dataset.pseudo_examples[i].label] for i in
                                range(len(train_dataset.pseudo_examples))]
                pseudo_sampler = RandomSampler(pseudo_data)
                pseudo_dataloader = DataLoader(pseudo_data, sampler=pseudo_sampler, batch_size=args.train_batch_size)
                model.train()
                weight = np.clip((idx * 2) / args.num_train_epochs, 0.0, 1.0).item()
                for step, pseudo_data in enumerate(pseudo_dataloader):
                    tokens = pseudo_data[0].to(args.device)
                    labels = pseudo_data[1].to(args.device)

                    loss, _ = model(tokens, labels, pseudo=True, weight=weight)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        try:
                            from apex import amp
                        except ImportError:
                            raise ImportError(
                                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    pseudo_loss.append(loss.item())
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                logger.info('Loss of pseudo label training %s', str(np.array(pseudo_loss).mean()))

            train_dataset.empty_pseudo()

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, model, tokenizer, checkpoint=str(args.start_epoch + idx))

            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            # model_to_save.save_pretrained(last_output_dir)
            torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, 'pytorch_model.bin'))
            logger.info("Saving model checkpoint to %s", last_output_dir)
            idx_file = os.path.join(last_output_dir, 'idx_file.txt')
            with open(idx_file, 'w', encoding='utf-8') as idxf:
                idxf.write(str(args.start_epoch + idx) + '\n')

            torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

            step_file = os.path.join(last_output_dir, 'step_file.txt')
            with open(step_file, 'w', encoding='utf-8') as stepf:
                stepf.write(str(global_step) + '\n')

            if (results['jaccard similarity score'] > best_acc):
                best_acc = results['jaccard similarity score']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                # model_to_save.save_pretrained(output_dir)
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev'):
    eval_outputs_dir = args.output_dir

    results = {}
    if not os.path.exists(eval_outputs_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_outputs_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = TextDataset(tokenizer, args, phase='validation')
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    # logger.info("begin evaluation (feed sample to NN)")
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            tokens = batch[0]
            labels = batch[1]
            tmp_eval_loss, logits = model(tokens, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    # logger.info("End evaluation ")
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds_label = np.clip(np.sign(preds), a_min=0, a_max=None)
        # preds_label = preds > args.threshold
    assert len(preds_label) == len(out_label_ids)

    # calculate accuracy of each lass
    # total_count = np.zeros(11)
    # true_count = np.zeros(11)
    # total_count += preds_label.shape[0]
    # true_count += (preds_label == out_label_ids).sum(axis=0)
    # print(true_count / total_count)
    result = jaccard_and_f1(preds_label, out_label_ids)
    results.update(result)
    if mode == 'dev' or mode == 'test':
        output_eval_file = os.path.join(eval_outputs_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            writer.write('evaluate %s\n' % checkpoint)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def jaccard_and_f1(preds, labels):
    jaccard = jaccard_score(preds, labels, average='samples')
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    f1_samples = f1_score(y_true=labels, y_pred=preds, average='samples')
    f1_each = f1_score(y_true=labels, y_pred=preds, average=None)
    return {
        "jaccard similarity score": jaccard,
        "f1_samples": f1_samples,
        "f1_micro": f1_micro,
        "f1_each": f1_each,
    }


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, tweet_tokens, tweet_ids, label):
        self.tweet_tokens = tweet_tokens
        self.tweet_ids = tweet_ids
        self.label = label


def convert_examples_to_features(item, tokenizer, args):
    # label
    label = [int(item[l]) for l in list(item.keys())[2:]]

    content = item['Tweet']
    line = normalizeTweet(content)

    tweet_tokens = tokenizer.tokenize(line)[:args.max_seq_length-2]
    tweet_tokens = [tokenizer.cls_token]+tweet_tokens+[tokenizer.sep_token]
    tweet_ids = tokenizer.convert_tokens_to_ids(tweet_tokens)
    padding_length = args.max_seq_length - len(tweet_ids)
    tweet_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(tweet_tokens, tweet_ids, label)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, phase=None, semi=False):
        dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english')
        dataset = dataset[phase]
        self.examples = []
        self.pseudo_examples = []

        for item in dataset:
            self.examples.append(convert_examples_to_features(item, tokenizer, args))

        if phase == 'train' and semi:
            random.shuffle(self.examples)
            self.unlabel_examples = self.examples[:args.unlabel_num]
            self.examples = self.examples[args.unlabel_num:]

        if phase == 'train':
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("tweet_tokens: {}".format([x.replace('\u0120', '_') for x in example.tweet_tokens]))
                logger.info("tweet_ids: {}".format(' '.join(map(str, example.tweet_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].tweet_ids), \
               torch.tensor(self.examples[i].label)
               # torch.tensor(i)

    def update_data(self, data, label):
        self.pseudo_examples.append(InputFeatures(None, data, label))

    def empty_pseudo(self):
        self.pseudo_examples = []


def parser_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str, required=False,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="The checkpoint path of model to continue training.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--train_file", default="train_top10_concat.tsv", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="test file")
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    parser.add_argument("--unlabel_num", default=5000, type=int,
                        help='the number of unlabelled data')
    parser.add_argument("--semi", action='store_true',
                        help='whether to use semi-supervised learning')
    parser.add_argument("--threshold", default=0.9, type=float,
                        help='the threshold to decide pseudo labels')
    parser.add_argument("--naivepl", action='store_true',
                        help='whether to calculate uncertainty')
    parser.add_argument("--pseudo_start", default=10, type=int,
                        help='the epoch to start using pseudo labels to train')
    parser.add_argument("--pos_certainty_threshold", default=0.05, type=float,
                        help='the uncertainty level threshold to use positive pseudo labels')
    parser.add_argument("--neg_certainty_threshold", default=0.005, type=float,
                        help='the uncertainty level threshold to use negative pseudo labels')
    parser.add_argument("--sharpen", action='store_true',
                        help='whether to use sharpened dynamic pseudo labeling')
    parser.add_argument("--onlyups", action='store_true',
                        help='only use UPS when doing semi-supervised learning')
    args = parser.parse_args()

    return args


class Model(nn.Module):
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.args = args

        self.linear = nn.Linear(768, 11)
        self.loss_func = nn.MultiLabelSoftMarginLoss()

    def forward(self, x, label, pseudo=False, weight=1.0, sharpen=False):
        x = self.encoder(x,  attention_mask=x.ne(1))[1]
        logits = self.linear(x)

        margin_loss = 0
        loss_count = 0
        if pseudo is False:
            if not sharpen:
                loss = self.loss_func(logits, label)
            else:
                loss = - (10 * torch.log(torch.sigmoid(logits)[label == 1]).sum() + \
                       torch.log(1 - torch.sigmoid(logits)[label == 0]).sum()) / (label.size(0) * label.size(1))
                for i in range(logits.size(0)):
                    if (label == 1)[i].any():
                        margin_loss += torch.sigmoid(logits[i])[(label == 1)[i]].mean() - \
                                       torch.sigmoid(logits[i])[(label == 0)[i]].mean()
                        loss_count += 1
                margin_loss = margin_loss / loss_count
                loss = loss - margin_loss
        elif pseudo is True:
            if (label == 1).any() and (label == 2).any():
                loss = - (torch.log(torch.sigmoid(logits)[label == 1] + 1e-6).mean() + torch.log(1 - torch.sigmoid(logits)[label == 2] + 1e-6).mean())
                # loss = - (torch.log(torch.sigmoid(logits)[label == 1]).sum() + \
                #    torch.log(1 - torch.sigmoid(logits)[label == 0]).sum()) / ((label == 1).sum() + (label == 2).sum())
                # for i in range(logits.size(0)):
                #     if (label == 1)[i].any() and (label == 2)[i].any():
                #         margin_loss += torch.sigmoid(logits[i])[(label == 1)[i]].mean() - torch.sigmoid(logits[i])[
                #             (label == 2)[i]].mean()
                #         loss_count += 1
                # loss = loss - margin_loss / loss_count
            elif (label == 1).any():
                loss = - torch.log(torch.sigmoid(logits)[label == 1] + 1e-6).mean()
            elif (label == 2).any():
                loss = - torch.log(1 - torch.sigmoid(logits)[label == 2] + 1e-6).mean()
            loss = weight * loss

        return loss, logits


def main():
    args = parser_arguments()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    args.output_mode = 'classification'
    num_labels = 2

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    model = Model(bertweet, args)

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    if args.checkpoint_path:
        logger.info(
            'Reload from {} using args.checkpoint_path'.format(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
        else:
            model.load_state_dict(
                torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location=torch.device('cuda:0')
        if torch.cuda.is_available() else torch.device('cpu')))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, phase='train', semi=args.semi)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, optimizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        # model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print(checkpoint)
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model_path = os.path.join(checkpoint, 'pytorch_model.bin')
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            model.to(args.device)
            result = evaluate(args, model, tokenizer, checkpoint=checkpoint, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_predict:
        print('testing')
        model_path = os.path.join(args.output_dir, 'checkpoint-best/pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        # print(model)
        evaluate(args, model, tokenizer, checkpoint=None, prefix='', mode='test')
    return results


if __name__ == "__main__":
    main()
