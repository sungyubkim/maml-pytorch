"""
This script evaluates a saved meta_learner (will crash if nothing is saved).
"""
import os
import time

import numpy as np
import scipy.stats as st
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
import arguments
from dataloader import MiniImagenet
from collections import OrderedDict


def evaluate(args, meta_learner, logger, dataloader, mode):

    for step, batch in enumerate(dataloader):

        # move episode to device
        for i in range(len(batch)):
            batch[i] = batch[i].to(args.device)
        support_x, support_y, query_x, query_y = batch
        meta_learner.eval()

        for inner_batch_idx in range(support_x.shape[0]):

            tuned_params = OrderedDict({})
            # only fine-tune the conv/fc weights/biases. do not fine-tuned the bn params
            for k, v in meta_learner.named_parameters():
                if ('conv' in k) or ('fc' in k):
                    tuned_params[k] = v.clone()
            # decoupling the base/meta learner makes faster 2nd order calc
            if args.decoupled:
                tuned_params= OrderedDict(
                    [(k,tuned_params[k]) 
                    for k in tuned_params.keys()
                    if 'fc' in k]
                    ) 

            logger.log_pre_update(support_x[inner_batch_idx], support_y[inner_batch_idx],
                                  query_x[inner_batch_idx], query_y[inner_batch_idx],
                                  meta_learner, mode)

            inner_iter = args.grad_steps_num_train if mode=='train' else args.grad_steps_num_eval
            for j in range(inner_iter):

                # get inner-grad
                in_pred = meta_learner(support_x[inner_batch_idx], tuned_params)
                in_loss = F.cross_entropy(in_pred, support_y[inner_batch_idx])
                in_grad = torch.autograd.grad(
                    in_loss,
                    tuned_params.values()
                )

                # update base-learner
                for k, g in zip(tuned_params.keys(), in_grad):
                    tuned_params[k] = tuned_params[k] - args.lr_in * g

            logger.log_post_update(support_x[inner_batch_idx], support_y[inner_batch_idx],
                                   query_x[inner_batch_idx], query_y[inner_batch_idx],
                                   meta_learner, tuned_params, mode)


class Logger:
    def __init__(self, args):

        self.args = args

        # initialise dictionary to keep track of accuracies/losses
        self.train_stats = {
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
        }
        self.valid_stats = {
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
        }
        self.test_stats = {
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
        }

        # keep track of how long the experiment takes
        self.start_time = time.time()

    def log_pre_update(self, support_x, support_y, query_x, query_y, meta_learner, mode):
        if mode == 'train':
            self.train_stats['train_accuracy_pre_update'].append(self.get_accuracy(support_x, support_y, meta_learner))
            self.train_stats['test_accuracy_pre_update'].append(self.get_accuracy(query_x, query_y, meta_learner))
        elif mode == 'val':
            self.valid_stats['train_accuracy_pre_update'].append(self.get_accuracy(support_x, support_y, meta_learner))
            self.valid_stats['test_accuracy_pre_update'].append(self.get_accuracy(query_x, query_y, meta_learner))
        elif mode == 'test':
            self.test_stats['train_accuracy_pre_update'].append(self.get_accuracy(support_x, support_y, meta_learner))
            self.test_stats['test_accuracy_pre_update'].append(self.get_accuracy(query_x, query_y, meta_learner))

    def log_post_update(self, support_x, support_y, query_x, query_y, meta_learner, tuned_params, mode):
        if mode == 'train':
            self.train_stats['train_accuracy_post_update'].append(self.get_accuracy(support_x, support_y, meta_learner, tuned_params))
            self.train_stats['test_accuracy_post_update'].append(self.get_accuracy(query_x, query_y, meta_learner, tuned_params))
        elif mode == 'val':
            self.valid_stats['train_accuracy_post_update'].append(self.get_accuracy(support_x, support_y, meta_learner, tuned_params))
            self.valid_stats['test_accuracy_post_update'].append(self.get_accuracy(query_x, query_y, meta_learner, tuned_params))
        elif mode == 'test':
            self.test_stats['train_accuracy_post_update'].append(self.get_accuracy(support_x, support_y, meta_learner, tuned_params))
            self.test_stats['test_accuracy_post_update'].append(self.get_accuracy(query_x, query_y, meta_learner, tuned_params))

    def print_header(self):
        print(
            '||------------------------------------------------------------------------------------------------------------------------------------------------------||')
        print(
            '||------------- TRAINING ------------------------|---------------------------------------- EVALUATION --------------------------------------------------||')
        print(
            '||-----------------------------------------------|------------------------------------------------------------------------------------------------------||')
        print(
            '||-----------------|     observed performance    |          META_TRAIN         |           META_VALID         |           META_TEST                     ||')
        print(
            '||    selection    |-----------------------------|-----------------------------|------------------------------|-----------------------------------------||')
        print(
            '||    criterion    |    train     |     valid    |    train     |     test     |    train     |     test      |    train     |     test                 ||')
        print(
            '||-----------------|--------------|--------------|--------------|--------------|--------------|---------------|--------------|--------------------------||')

    def print_logs(self, selection_criterion, logged_perf=None):
        if logged_perf is None:
            logged_perf = [' ', ' ']
        else:
            logged_perf = [np.round(logged_perf[0], 3), np.round(logged_perf[1], 3)]

        avg_acc = np.mean(self.test_stats['test_accuracy_post_update'])
        conf_interval = st.t.interval(0.95,
                                      len(self.test_stats['test_accuracy_post_update']) - 1,
                                      loc=avg_acc,
                                      scale=st.sem(self.test_stats['test_accuracy_post_update']))

        print(
            '||   {:<11}   |    {:<5}     |     {:<5}    | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5}  | {:<5}->{:<5} | {:<5}->{:<5} (+/- {}) ||'.format(
                selection_criterion,
                # performance we observed during training
                logged_perf[0],
                logged_perf[1],
                # meta-valid, task-test
                np.round(np.mean(self.train_stats['train_accuracy_pre_update']), 3),
                np.round(np.mean(self.train_stats['train_accuracy_post_update']), 3),
                np.round(np.mean(self.train_stats['test_accuracy_pre_update']), 3),
                np.round(np.mean(self.train_stats['test_accuracy_post_update']), 3),
                #
                np.round(np.mean(self.valid_stats['train_accuracy_pre_update']), 3),
                np.round(np.mean(self.valid_stats['train_accuracy_post_update']), 3),
                np.round(np.mean(self.valid_stats['test_accuracy_pre_update']), 3),
                np.round(np.mean(self.valid_stats['test_accuracy_post_update']), 3),
                #
                np.round(np.mean(self.test_stats['train_accuracy_pre_update']), 3),
                np.round(np.mean(self.test_stats['train_accuracy_post_update']), 3),
                np.round(np.mean(self.test_stats['test_accuracy_pre_update']), 2),
                np.round(100 * avg_acc, 3),
                #
                np.round(100 * np.mean(np.abs(avg_acc - conf_interval)), 2),
            ))

    def get_accuracy(self, x, y, meta_learner, tuned_params=None):
        with torch.no_grad():
            meta_learner.eval()
            predictions = meta_learner(x, tuned_params)
            num_correct = torch.argmax(F.softmax(predictions, dim=1), 1).eq(y).sum().item()
        return num_correct / len(y)


if __name__ == '__main__':

    args = arguments.parse_args()

    # save path
    path = 'results/'+args.backbone
    if args.decoupled:
        path += '_decoupled'
    else:
        path += '_coupled'
    if args.first_order:
        path += '_first_order'
    else:
        path += '_second_order'

    # initialise logger
    logger = Logger(args)
    logger.print_header()

    # initialise logger
    logger = Logger(args)

    for selection_criterion in ['valid']:

        logger = Logger(args)

        for dataset in ['train', 'val', 'test']:
            # load meta_learner and its performance during training
            meta_learner = torch.load(path + '_best_{}'.format(selection_criterion)).to(args.device)
            best_performances = np.load(path + '_best_{}.npy'.format(selection_criterion))

            # initialise dataloader
            mini_test = MiniImagenet(mode=dataset, n_way=args.n_way,
                                        k_shot=args.k_shot, k_query=args.k_query,
                                        batchsz=500, verbose=True, imsize=84, data_path=args.data_path)
            db_test = DataLoader(mini_test, batch_size=args.batch_size, shuffle=True, num_workers=0)

            # evaluate the meta_learner
            evaluate(args, meta_learner, logger, db_test, mode=dataset)

        logger.print_logs(selection_criterion, best_performances)
