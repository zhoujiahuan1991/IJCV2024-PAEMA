from __future__ import print_function, absolute_import
import time

import torch.nn as nn

from .utils.meters import AverageMeter
from .utils.ptkp_tools import *
from reid.metric_learning.distance import cosine_similarity, cosine_distance
from reid.utils.make_loss import make_loss


class Trainer(object):
    def __init__(self,cfg,args, model, num_classes,margin=0,ema=0 ,writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        
    def train(self, epoch, data_loader_train, optimizer, optimizer_prompt, training_phase,
              train_iters=200, add_num=0, old_model=None, replay=False):

        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()


        end = time.time()

        for i in range(train_iters):

            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num

            s_features, bn_features, s_cls_out, fake_feat_list,dis = self.model(s_inputs, domains, training_phase,epoch=epoch)


            loss_ce, loss_tp = self.loss_fn(s_cls_out, s_features, targets, target_cam=None)

            
            loss = loss_ce + loss_tp 

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            
            optimizer.zero_grad()
            optimizer_prompt.zero_grad()
            loss.backward()
            
            

            optimizer.step()
            optimizer_prompt.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,))
            #return


    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        return loss_ce, loss_tr

