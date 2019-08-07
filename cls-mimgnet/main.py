from models import Network
from arguments import parse_args
from dataloader import MiniImagenet
from logger import Logger
import utils

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

import time
from copy import deepcopy
from collections import OrderedDict

def inner_loop(args, meta_learner, support_x, support_y, query_x, query_y, logger, iter_counter, mode):
    '''
    run a single episode == n-way k-shot problem
    '''
    meta_learner.zero_grad()
    tuned_params = OrderedDict({})
    for k, v in meta_learner.named_parameters():
        tuned_params[k] = v

    logger.log_pre_update(iter_counter,
    support_x,
    support_y,
    query_x,
    query_y,
    meta_learner,
    mode=mode)

    inner_iter = args.grad_steps_num_train if mode=='train' else args.grad_steps_num_eval
    for j in range(inner_iter):

        # get inner-grad
        in_pred = meta_learner(support_x, tuned_params)
        in_loss = F.cross_entropy(in_pred, support_y)
        in_grad = torch.autograd.grad(
            in_loss,
            tuned_params.values(),
            create_graph=not(args.first_order)
        )

        # update base-learner
        for k, g in zip(tuned_params.keys(), in_grad):
            tuned_params[k] = tuned_params[k] - args.lr_in * g

    # get outer-grad
    out_pred = meta_learner(query_x, tuned_params)
    out_loss = F.cross_entropy(out_pred, query_y)
    out_grad = torch.autograd.grad(
        out_loss,
        meta_learner.parameters()
    )

    logger.log_post_update(iter_counter,
    support_x,
    support_y,
    query_x,
    query_y,
    meta_learner,
    tuned_params=tuned_params,
    mode=mode)

    meta_learner.zero_grad()

    return in_grad, out_grad

def outer_loop(args, meta_learner, opt, batch, logger, iter_counter):
    '''
    run a single batch == multiple episodes
    '''

    # move episode to device
    for i in range(len(batch)):
        batch[i] = batch[i].to(args.device)
    support_x, support_y, query_x, query_y = batch
    grad = [0. for p in meta_learner.parameters()]

    results = list()
    for i in range(args.batch_size):

        # accumulate grad to meta-learner using inner loop
        _, out_grad = inner_loop(args, 
        meta_learner, 
        support_x[i], 
        support_y[i],
        query_x[i],
        query_y[i],
        logger,
        iter_counter,
        mode='train')

        for i in range(len(out_grad)):
            grad[i] += out_grad[i].detach()

    meta_learner.zero_grad()
    for p, g in zip(meta_learner.parameters(), grad):
        p.grad = g/float(args.batch_size)

    # summarise inner loop and get validation performance
    logger.summarise_inner_loop(mode='train')

    torch.nn.utils.clip_grad_value_(meta_learner.parameters(), 10.0)

    opt.step()

    return None

def train(args, meta_learner, opt, dataloaders, logger):

    dataloader_train, dataloader_valid = dataloaders

    iter_counter = 0
    while iter_counter < args.n_iter:
        
        # iterate over epoch
        logger.print_header()

        for step, batch in enumerate(dataloader_train):

            logger.prepare_inner_loop(iter_counter)

            outer_loop(args, meta_learner, opt, batch, logger, iter_counter)

            # log/ save
            if (iter_counter % args.log_interval == 0):
                valid(args, meta_learner, dataloader_valid, logger,iter_counter)
                if args.save_path is not None:
                    np.save(args.save_path, [logger.training_stats, logger.validation_stats])
                    # save model to CPU
                    save_model = meta_learner
                    if args.device == 'cuda':
                        save_model = deepcopy(meta_learner).to('cpu')
                    torch.save(save_model, args.save_path)

            iter_counter += 1

    return None

def valid(args, meta_learner, dataloader_valid, logger, iter_counter):
    logger.prepare_inner_loop(iter_counter, mode='valid')

    for step, batch in enumerate(dataloader_valid):

        # move episode to device
        for i in range(len(batch)):
            batch[i] = batch[i].to(args.device)
        support_x, support_y, query_x, query_y = batch
        meta_learner.zero_grad()

        for i in range(support_x.shape[0]):

            # accumulate grad to meta-learner using inner loop
            in_grad, out_grad = inner_loop(args, 
            meta_learner, 
            support_x[i], 
            support_y[i],
            query_x[i],
            query_y[i],
            logger,
            iter_counter,
            mode='valid')

    # this will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # keep track of best models
    logger.update_best_model(meta_learner, args.save_path)

    # print the log
    logger.print(iter_counter, in_grad, out_grad, mode='valid')

    return None

def run(args):

    utils.set_seed(args.seed)

    # make nets
    meta_learner = Network(n_channel=args.n_channel,
                            n_way=args.n_way).to(args.device)

    # make optimizer
    opt = torch.optim.Adam(meta_learner.parameters(), args.lr_out)

    # make datasets/ dataloaders
    dataset_train = MiniImagenet(mode='train',
                                    n_way=args.n_way,
                                    k_shot=args.k_shot,
                                    k_query=args.k_query,
                                    batchsz=10000,
                                    imsize=84,
                                    data_path=args.data_path)
    dataloader_train = DataLoader(dataset_train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    dataset_valid = MiniImagenet(mode='val',
                                    n_way=args.n_way,
                                    k_shot=args.k_shot,
                                    k_query=args.k_query,
                                    batchsz=100,
                                    imsize=84,
                                    data_path=args.data_path)
    dataloader_valid = DataLoader(dataset_valid,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

    dataloaders = (dataloader_train, dataloader_valid)

    # make logger
    logger = Logger(args)

    # train nets
    train(args, meta_learner, opt, dataloaders, logger)

    # write results
    return None

if __name__=='__main__':
    args = parse_args()
    run(args)