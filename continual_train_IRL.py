from __future__ import print_function, absolute_import
import argparse
from ast import arg
import os.path as osp
import sys

from torch.backends import cudnn
import copy
import torch.nn as nn
import random
from config import cfg
from reid import datasets
from reid.evaluators import Evaluator,Evaluator_twomodel
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict,copy_state_dict_save_prompt,copy_state_dict_save_param
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.ptkp_tools import *

from reid.models.resnet import build_resnet_backbone
from reid.models.layers import DataParallel
from reid.models.vit import make_model
from reid.trainer import Trainer
import json
from torch.utils.tensorboard import SummaryWriter
from reid.datasets.get_data_loaders import build_data_loaders
from multiprocessing import process


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    main_worker(args,cfg)


def main_worker(args,cfg):

    cudnn.benchmark = True
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    training_set = ['market1501', 'dukemtmc', 'cuhk_sysu', 'msmt17']
    #training_set = ['msmt17']
    if args.test :
        all_set = ['market1501', 'dukemtmc', 'cuhk_sysu', 'msmt17', 'cuhk01', 'cuhk03', 'grid','sense']  # 'sense'
    else :
        all_set = ['market1501', 'dukemtmc', 'cuhk_sysu', 'msmt17']  # 'sense'
    #all_set = ['msmt17']
    testing_only_set = [x for x in all_set if x not in training_set]

    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)

    # Create model
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num = 0,cfg=cfg)
    model.cuda()
    model = DataParallel(model)

    writer = SummaryWriter(log_dir=args.logs_dir)
    # Load from checkpoint
    if args.resume:
        
        checkpoint = load_checkpoint(args.resume)
        #print(checkpoint['prompt_selected_sum_train'])
        #print(checkpoint['prompt_selected_train'])
        #exit(0)
        copy_state_dict(checkpoint['state_dict'], model)
        #copy_state_dict_save_prompt(checkpoint['state_dict'], model)
        #copy_state_dict_save_param(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
    
    if args.test :
        ckpt_name = ['market1501_checkpoint.pth.tar','dukemtmc_checkpoint.pth.tar','cuhk_sysu_checkpoint.pth.tar','msmt17_checkpoint.pth.tar']
        print('------------first model ----------')
        checkpoint = load_checkpoint(osp.join( args.resume_EMA,ckpt_name[0]))
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
        
        for step in range(3):
            

            model_old=copy.deepcopy(model)

            print('------------second model ----------')
            checkpoint = load_checkpoint(osp.join( args.resume_EMA,ckpt_name[step+1]))
            copy_state_dict(checkpoint['state_dict'], model)
            start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['mAP']
            print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
            
            
            print('------------EMA model ----------')
            best_alpha = get_adaptive_alpha(args,model,model_old)
            
            print('********combining new model and old model with alpha {}********\n'.format(best_alpha))
            model=linear_combination(args,model,model_old, best_alpha)#,model_old_id=step+2)

                
        
        test_model(model, all_train_sets, all_test_only_sets, 3)
        return 
    
    # Evaluator
    out_channel = 768


    print('training the first dataset: {}'.format(training_set[0]))
    save_checkpoint({
        'state_dict': model.state_dict(),
        'epoch': 0,
        'mAP': 0,
    }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format('begin')))

    model = train_first(cfg,args, first_train_set, model)

    if args.train_first :
        print('train first done')
        return 

    for set_index in range(1,len(training_set)):
        model_old=copy.deepcopy(model)
        model = train_next(cfg,args,all_train_sets,all_test_only_sets,set_index,model,out_channel,writer)


        best_alpha = get_adaptive_alpha(args,model,model_old)
        print('********combining new model and old model with alpha {}********\n'.format(best_alpha))
        model=linear_combination(args,model,model_old, best_alpha)
        # # print_model_param(model, name='combined_model')
        test_model(model, all_train_sets, all_test_only_sets, set_index)
        

    print('finished')



def train_first(cfg,args, first_train_set,model):
    
    


    evaluator = Evaluator(model)
    dataset, num_classes, train_loader, test_loader, init_loader, name=first_train_set
    # Opitimizer initialize
    params = []
    params_prompt = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            #print(key,'????')
            continue

        if key in ['module.base.pool.key_list','module.base.pool.prompt_list','module.base.general_prompt','module.base.dirty_prompt_param']:
            params_prompt += [{"params": [value], "lr": args.lr2}]
            #print(key,'1111')
        else :
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
            #print(key,'222')
    if args.optimizer =='Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer =='SGD':
        optimizer = torch.optim.SGD(params,momentum=args.momentum)
    if args.optimizer2 =='Adam':
        optimizer_prompt = torch.optim.Adam(params_prompt)
    elif args.optimizer2 =='SGD':
        optimizer_prompt = torch.optim.SGD(params_prompt,momentum=args.momentum)
    lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    lr_scheduler_prompt = WarmupMultiStepLR(optimizer_prompt, [40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
    trainer = Trainer(cfg,args,model, num_classes)
    epoch=0

    for epoch in range(0, args.epochs0):
        # evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer,optimizer_prompt, training_phase=1,
                      train_iters=len(train_loader), add_num=0, old_model=None, replay=False)
        lr_scheduler.step()
        lr_scheduler_prompt.step()

        if ((epoch + 1) % args.eval_epoch== 0):
            print('Results on {}'.format('Market'))
            mAP=test_model(model, [first_train_set],[], 0)

            save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch+1,
                    'mAP': mAP,
                }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))
    

    return model


def train_next(cfg,args, all_train_sets, all_test_only_sets,set_index, model,out_channel,writer):
    dataset, num_classes, train_loader, test_loader, init_loader, name=all_train_sets[set_index]  # status of current dataset


    if 1==set_index:
        add_num=0
    else:
        add_num = sum([all_train_sets[i][1] for i in range(set_index-1)])  
    
    dataset_last, _, _, _, _, last_name = all_train_sets[set_index - 1]

    old_model = None

    add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
    # Expand the dimension of classifier
    org_classifier_params = model.module.classifier.weight.data
    print(org_classifier_params.shape)
    model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
    model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
    model.cuda()

    # Initialize classifer with class centers
    #print('---warning: not Initialize classifer with class centers')
    
    class_centers = initial_classifier(model, init_loader)
    model.module.classifier.weight.data[add_num:].copy_(class_centers)
    model.cuda()
    
    # Create old frozen model
    

    print('-------Task {} classifier stored------------'.format(set_index))

    
    # Re-initialize optimizer
    params = []
    params_prompt = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:',key)
            continue
        if key in ['module.base.pool.key_list','module.base.pool.prompt_list','module.base.general_prompt','module.base.dirty_prompt_param']:
            params_prompt += [{"params": [value], "lr": args.lr2}]
        else :
            params += [{"params": [value], "lr": args.lr*args.lrscale , "weight_decay": args.weight_decay}]
    if args.optimizer =='Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer =='SGD':
        optimizer = torch.optim.SGD(params,momentum=args.momentum)
    if args.optimizer2 =='Adam':
        optimizer_prompt = torch.optim.Adam(params_prompt)
    elif args.optimizer2 =='SGD':
        optimizer_prompt = torch.optim.SGD(params_prompt,momentum=args.momentum)
    Stones = [20, 30] if name == 'msmt17' else [30]
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    lr_scheduler_prompt = WarmupMultiStepLR(optimizer_prompt, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)


    trainer = Trainer(cfg,args,model, add_num + num_classes,writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, args.epochs):

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer,optimizer_prompt, training_phase=set_index+1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model, replay=False)
        
        
        lr_scheduler.step()
        lr_scheduler_prompt.step()
        
        if ((epoch + 1) % args.eval_epoch== 0):

            #mAP=test_model(model, all_train_sets,all_test_only_sets, set_index)
            if set_index == 3 :
                mAP=test_model(model, all_train_sets,all_test_only_sets, set_index-1)
            else :
                mAP=test_model(model, all_train_sets,all_test_only_sets, set_index)

            save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'mAP': mAP,
                }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

    return model


def test_model(model, all_train_sets, all_test_sets, set_index):
    begin=0
    evaluator = Evaluator(model)

    R1_all = []
    mAP_all = []
    for i in range(begin,set_index + 1):

        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))
        train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                                 cmc_flag=True)  # ,training_phase=i+1)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()
    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))


    R1_all = []
    mAP_all = []
    for i in range(len(all_test_sets)):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_test_sets[i]
        print('Results on {}'.format(name))
        R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                     cmc_flag=True)

        R1_all.append(R1)
        mAP_all.append(mAP)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1 * 100))
    return train_mAP


def linear_combination(args,model, model_old, alpha):

    '''old model '''
    model_old = model_old.to('cpu')
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model = model.to('cpu')
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()

    for k,v in model_state_dict.items():
        

        if k in['module.base.dirty_prompt_param','module.base.pool.key_list','module.base.pool.prompt_list']:
            model_new_state_dict[k]=v
            model_new_state_dict[k]=model_old_state_dict[k]
        elif model_old_state_dict[k].shape==v.shape:
            model_new_state_dict[k]=alpha*v+(1-alpha)*model_old_state_dict[k]
        else:
            num_class_old=model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]

    model_new.load_state_dict(model_new_state_dict)
    model_new = model_new.cuda()
    return model_new

def get_adaptive_alpha(args,model, model_old):
    print('enter get_adaptive_alpha')

    '''old model '''
    model_old = model_old.to('cpu')
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model = model.to('cpu')
    model_state_dict = model.state_dict()

    dif_list = []
    changing_param = []
    for k, v in model.module.named_parameters(): # only consider trainable parameters
        k = 'module.' + k
        if v.requires_grad:   # escape parameters about prompt
            if k == 'module.base.general_prompt' :
                dif = model_state_dict[k] - model_old_state_dict[k]
                print(k)
                ratio = 2 * torch.abs(dif) / (torch.abs(model_old_state_dict[k]) + torch.abs(model_state_dict[k]))
                ''' relative change compared to old parameter'''
                #ratio = torch.abs(dif) / torch.abs(model_old_state_dict[k])
                print(torch.mean(ratio,dim=(1,2)))
                dif_list.append(ratio.mean())
                changing_param.append(k)
                break
            
    print(len(dif_list))
    ratio=torch.stack(dif_list, dim=0).mean()
    print('ema ratio: ',ratio,'------------------')

    c_alpha = ratio
    return c_alpha





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--MODEL', type=str, default='dual_transformer2',choices=['50x','transformer','transformer_jmp','dual_transformer','dual_transformer2'])
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD',choices=['SGD','Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--lrscale', type=float, default=1,
                        help="learning rate of new parameters, for pretrained ")
                        
    parser.add_argument('--optimizer2', type=str, default='Adam',choices=['SGD','Adam'],
                        help="optimizer2 ")
    parser.add_argument('--lr2', type=float, default=0.005,
                        help="learning rate of new parameters, for frozen model ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=20)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # path
    this_path = osp.dirname(osp.abspath(__file__))
    #parser.add_argument('--data-dir', type=str, metavar='PATH',
    #                    default=osp.join(working_dir, 'data'))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data/dataset/xukunlun/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs_vit'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")

    parser.add_argument('--config_file', type=str, default='configs/Market/vit_base.yml',
                        help="config_file")
    parser.add_argument('--test', default=False, action='store_true',help="test the trained model")
    parser.add_argument('--exemplar', default=False, action='store_true',help="exemplar")
    parser.add_argument('--test_EMA', default=False, action='store_true',help="test EMA")
    parser.add_argument('--alpha', type=float, default=0.7,help="EMA update")
    parser.add_argument('--prompt_type', type=int, default=0,help="methods to use prompt")
    parser.add_argument('--prompt_num', type=int, default=20,help="prompt_num")
    parser.add_argument('--topN', type=int, default=4,help="prompt_num")
    parser.add_argument('--pool_size', type=int, default=4,help="task num")
    parser.add_argument('--gprompt_num', type=int, default=4,help="gprompt_num")
    parser.add_argument('--fuse', default=False, action='store_true',help="fuse prompt")
    parser.add_argument('--key', default=False, action='store_true',help="use key for last dataset")
    parser.add_argument('--layer_g', type=int,nargs='*', required=True,help="layer gprompt")
    parser.add_argument('--layer_e', type=int,nargs='*', required=True,help="layer eprompt")
    parser.add_argument('--prompt_init_type', type=str, default='unif',choices=['default','unif'],help="prompt_init_type")
    parser.add_argument('--key_layer', type=int, default=12,help="the block num to optimize key")
    parser.add_argument('--resume_EMA', type=str, default=None,help="log folder to resume EMA")
    parser.add_argument('--resume_folder', type=str, default=None,help="log folder to resume EMA2")
    parser.add_argument('--gprompt_init_type', type=str, default='unif',choices=['default','default0.5','unif','zeros','onenum_0.5','default0.5_0.25'],help="prompt_init_type")
    parser.add_argument('--train_first', default=False, action='store_true',help="fuse prompt")
    main()