# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

import pandas as pd

import logging
import os,sys
import argparse
import random
from tqdm import tqdm, trange
#from run_copa import do_evaluation
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice,BertForMultipleChoiceMarginLoss
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from whim.entity_narrative.eval.multiple_choice import MultipleChoiceQuestion

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class MCNCEventExample(object):
    """A single training/test example for the roc dataset."""
    def __init__(self, mcq):
        self.context_sentence = mcq.context_events
        self.endings = mcq.choices
        self.entity = mcq.entity
        self.document = mcq.document.to_text()

        self.label = mcq.target_choice

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.document.entities.index(self.entity)}",
            f"entity: {self.entity}",
            f"context_sentence: {''.join([e.to_text(self.document.entities) for e in self.context_events])}",
            f"ending1: {self.endings[0]}",
            f"ending2: {self.endings[1]}",
            f"ending3: {self.endings[2]}",
            f"ending4: {self.endings[3]}",
            f"ending5: {self.endings[4]}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)





class InputFeatures(object):
    def __init__(self,
                 # example_id,
                 choices_features,
                 label

    ):
        # self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_mcnc_examples(input_dir, is_training):

    # Read in pre-prepared questions
    print("Reading in %d questions from %s" % (len(os.listdir(input_dir)), input_dir))
    examples = []
    for filename in os.listdir(input_dir):
        print(filename)
        with open(os.path.join(input_dir, filename), 'r',encoding='utf-8') as question_file:
            examples.append(MCNCEventExample(MultipleChoiceQuestion.from_text(question_file.read())))

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # roc is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given roc example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        # if example_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info(f"roc_id: {example.roc_id}")
        #     for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
        #         logger.info(f"choice: {choice_idx}")
        #         logger.info(f"tokens: {' '.join(tokens)}")
        #         logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
        #         logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
        #         logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
        #     if is_training:
        #         logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def write_result_to_file(args,result):
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(result+"\n")

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def predict_on_test(model,eval_dataloader,args,epoch,best_eval_acc):
    output_csv=pd.read_csv(os.path.join(args.data_dir, 'output_testnew.csv'))
    model.eval()
    logits_all=None
    for input_ids, input_mask, segment_ids, _ in eval_dataloader:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        if logits_all is None:
            logits_all=logits.copy()
        else:
            logits_all=np.vstack((logits_all,logits))
    answer = np.argmax(logits_all, axis=1)
    for index, data in enumerate(answer):
        output_csv['AnswerRightEnding'][index]=int(data+1)
    model_name_list=args.bert_model.split('/')
    model_name=model_name_list[-2]+'_'+model_name_list[-1]
    if len(model_name_list)>=2:
        model_name=model_name_list[-2]+'_'+model_name_list[-1]
    else:
        model_name=args.bert_model
    output_csv.to_csv(os.path.join(args.data_dir, f'answer_testnew_{best_eval_acc}_{epoch}_{args.do_margin_loss}_{args.margin}_{args.seed}_{model_name}.txt'),index=False,float_format='%.f')


def do_evaluation(model,eval_dataloader,args,is_training=False):
    if is_training:
        eval_flag='train'
    else:
        eval_flag='eval'
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all=None
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            if not args.no_cuda:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                label_ids = label_ids.cuda()
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            if logits_all is None:
                logits_all=logits.copy()
            else:
                logits_all=np.vstack((logits_all,logits))
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {f'{eval_flag}_loss': eval_loss,
              'seed': args.seed,
              f'{eval_flag}_accuracy': eval_accuracy,}
              # 'global_step': global_step,
              # 'loss': tr_loss/nb_tr_steps}
    # logger.info("  %s = %s", f'{eval_flag}_accuracy', str(result[f'{eval_flag}_accuracy']))
    # logger.info(f"***** {eval_flag} results *****")
    # for key in sorted(result.keys()):
    #     logger.info("  %s = %s", key, str(result[key]))
    model.zero_grad()
    return logits_all,eval_accuracy
    # write_result_to_file(args,result)


def do_evaluation_back_up(model,eval_dataloader,args):
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'eval_loss': eval_loss,
                  'seed': args.seed,
                  'eval_accuracy': eval_accuracy,}
                  # 'global_step': global_step,
                  # 'loss': tr_loss/nb_tr_steps}
        # logger.info("  %s = %s", 'eval_accuracy', str(result['eval_accuracy']))
        # logger.info("***** Eval results *****")
        # for key in sorted(result.keys()):
        #     logger.info("  %s = %s", key, str(result[key]))
        print("eval_accuracy:",eval_accuracy)                    
        # write_result_to_file(args,result)

def main(MARGIN=0.15):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The train dataset folder. it should contains .txt files (or other data files) for the task.")
    parser.add_argument("--dev_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The dev dataset folder. it should contains .txt files (or other data files) for the task.")
    parser.add_argument("--test_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The test dataset folder. it should contains .txt files (or other data files) for the task.")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=40,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--do_margin_loss",
                        default=0,
                        type=int,
                        help="Use margin loss or log-loss.")
    parser.add_argument("--learning_rate",
                        default=6.25e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--margin',
                        type=float, default=0.15,
                        help='Margin value used in the MultiMarginLoss.')
    # parser.add_argument('--gpuid', type=int, default=-1,help='The gpu id to use')

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        if not args.no_cuda:
            # device = torch.device("cuda",args.gpuid)
            # torch.cuda.set_device(args.gpuid)
            dummy=torch.cuda.FloatTensor(1)
        else:
            device = torch.device("cpu")
        n_gpu = 1
        # n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    # logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_prediction:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_prediction` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    eval_examples = None
    num_train_steps = None
    eval_size=0
    if args.do_train:
        # train_examples = read_roc_examples(os.path.join(args.data_dir, 'val.csv'), is_training = True)
        train_examples = read_mcnc_examples(args.dataset_dir, is_training = True)
        # train_examples = read_roc_examples(os.path.join(args.data_dir, 'val_test.csv'), is_training = True)
        # train_examples = read_roc_examples(os.path.join(args.data_dir, 'val_test_valnew.csv'), is_training = True)
        if args.do_eval:
            eval_examples = read_mcnc_examples(args.dev_dir, is_training = False)
            # train_examples=train_examples+eval_examples[0:1000]
            # eval_examples=eval_examples[1000:]
            eval_size=len(eval_examples)
            print(len(train_examples),len(eval_examples))
            num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        else:
            num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.do_margin_loss==0:
        model = BertForMultipleChoice.from_pretrained(args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
            num_choices = 5
        )
    elif args.do_margin_loss==1:
        model = BertForMultipleChoiceMarginLoss.from_pretrained(args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
            num_choices = 5, margin=args.margin
        )

    if args.fp16:
        model.half()
    if not args.no_cuda:
        model.cuda()
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # eval_examples = read_roc_examples(os.path.join(args.data_dir, 'valnew.csv'), is_training = True)
        # eval_examples = read_roc_examples(os.path.join(args.data_dir, 'test.csv'), is_training = True)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
        # logger.info("***** Running evaluation *****")
        # logger.info("  Num examples = %d", len(eval_examples))
        args.eval_batch_size=args.train_batch_size
        # logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # do_evaluation(model,eval_dataloader,args)

    if args.do_prediction and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_examples =read_mcnc_examples(args.test_dir, is_training = False)
        test_features = convert_examples_to_features(
            test_examples, tokenizer, args.max_seq_length, True)
        # logger.info("***** Running testuation *****")
        # logger.info("  Num examples = %d", len(test_examples))
        # logger.info("  Batch size = %d", args.test_batch_size)
        all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        # do_evaluation(model,eval_dataloader,args)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        # logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_examples))
        # logger.info("  Batch size = %d", args.train_batch_size)
        # logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        best_eval_acc=0.0
        best_step=0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        # for epoch in range(int(args.num_train_epochs)):
            print("Epoch:",epoch)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.cuda() if not args.no_cuda else t for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if step % 100 == 0:
                    logging.info('Batch %d, loss=%.4f' % (step, loss))
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
                    if args.do_eval and epoch>0:
                        logits_all,eval_accuracy=do_evaluation(model,eval_dataloader,args,is_training=False)
                        if best_eval_acc<eval_accuracy:
                            best_eval_acc=eval_accuracy
                            best_step=global_step
                            if args.do_prediction:
                                predict_on_test(model,test_dataloader,args,epoch,best_eval_acc)
                            print(best_eval_acc)
                            # torch.save(model.state_dict(), os.path.join(args.output_dir, f"pytorch_model.bin"))
            model_save_dir = os.path.join(args.output_dir, f'model{epoch}')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"pytorch_model.bin"))
            print("model has been saved to " + model_save_dir + " bingo!!")
        print(best_eval_acc,args.seed,args.margin,args.do_margin_loss,args.learning_rate,args.bert_model,sys.argv[0])
        result=f"best_eval_acc={best_eval_acc},args.seed={args.seed},args.do_margin_loss={args.do_margin_loss},args.margin={args.margin},best_step={best_step},train_batch_size={args.train_batch_size},eval_size={eval_size},script_name={sys.argv[0]},model={args.bert_model},learning_rate={args.learning_rate},num_train_epochs={args.num_train_epochs},max_seq_length={args.max_seq_length}"
        write_result_to_file(args,result)

    if args.do_prediction:
        _, test_accuracy = do_evaluation(model, test_dataloader, args, is_training=False)
        print("test accurary is :" + str(test_accuracy))


if __name__ == "__main__":
    main()
