import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
import math
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
#from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from reid.utils.ptkp_tools import get_pseudo_features
from reid.models.gem_pool import GeneralizedMeanPoolingP
from .backbones.vit_dual_pytorch import  vit_base_patch16_224_TransReID_dual, vit_small_patch16_224_TransReID_dual, deit_small_patch16_224_TransReID_dual

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self,last_stride, bn_norm, with_ibn, with_se,block, num_classes,layers):
        super(Backbone, self).__init__()



        self.in_planes = 2048
        self.base = ResNet(last_stride=last_stride,
                            block=block,
                            layers=layers)
        print('using resnet50 as a backbone')
        

        

        self.bottleneck = nn.BatchNorm2d(2048)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.classifier = nn.Linear(512*block.expansion, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.task_specific_batch_norm = nn.ModuleList(nn.BatchNorm2d(512*block.expansion) for _ in range(5))
        print(512*block.expansion,'------------------')
        for bn in self.task_specific_batch_norm:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)

        self.random_init()
        self.num_classes = num_classes
        '''
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        '''
    def forward(self, x, domains=None, training_phase=None, disti=False, fkd=False):  
        
        x = self.base(x)
        global_feat = self.pooling_layer(x) # [16, 2048, 1, 1]
        bn_feat = self.bottleneck(global_feat) # [16, 2048, 1, 1]
        #if disti is True:
        #    prob = self.classifier(bn_feat[..., 0, 0])
        #    return global_feat[..., 0, 0], bn_feat[..., 0, 0], prob

        if fkd is True:
            fake_feat_list = get_pseudo_features(self.task_specific_batch_norm, training_phase,
                                                     global_feat, domains, unchange=True)
            cls_outputs = self.classifier(bn_feat[..., 0, 0])
            return global_feat[..., 0, 0], bn_feat[..., 0, 0], cls_outputs, fake_feat_list

        if self.training is False:
            return bn_feat[..., 0, 0]

        bn_feat = bn_feat[..., 0, 0]
        cls_outputs = self.classifier(bn_feat)

        fake_feat_list = get_pseudo_features(self.task_specific_batch_norm, training_phase, global_feat, domains)

        return global_feat[..., 0, 0], bn_feat, cls_outputs, fake_feat_list

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg,args, factory):
        super(build_transformer, self).__init__()

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.args = args
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](args =args,img_size=[256, 128], sie_xishu=3.0,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        #if pretrain_choice == 'imagenet':
        #    self.base.load_param(model_path)
        #    print('Loading pretrained ImageNet model......from {}'.format(model_path))

        

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        #self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        if self.args.exemplar :
            self.task_specific_batch_norm = nn.ModuleList(nn.BatchNorm1d(self.in_planes) for _ in range(5))

            for bn in self.task_specific_batch_norm:
                bn.bias.requires_grad_(False)
                nn.init.constant_(bn.weight, 1)
                nn.init.constant_(bn.bias, 0)

        self.random_init_laiming()
        '''
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        '''

    def random_init_laiming(self):

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias != None :
                    nn.init.constant_(m.bias, 0.0)

            elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('BatchNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x,domains=None, training_phase=None, disti=False, fkd=False, cam_label= None, view_label=None,epoch=0,old=False):
        #global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        if training_phase!=None:
            global_feat,dis = self.base(x,training_phase=training_phase-1,epoch=epoch)
        else :
            global_feat,dis = self.base(x,training_phase=training_phase,epoch=epoch)
        bn_feat = self.bottleneck(global_feat)
        if fkd is True:
            fake_feat_list = get_pseudo_features(self.task_specific_batch_norm, training_phase,
                                                     global_feat, domains, unchange=True)
            cls_outputs = self.classifier(bn_feat)
            return global_feat, bn_feat, cls_outputs, fake_feat_list
        if old is True:

            cls_outputs = self.classifier(bn_feat)
            return global_feat, bn_feat, cls_outputs, 0
        cls_outputs = self.classifier(bn_feat)
        if self.training is False:
            if self.args.key :
                return bn_feat,dis
            else :
                return bn_feat#[..., 0, 0]

        #bn_feat = bn_feat[..., 0, 0]
        
        if self.args.exemplar :
            fake_feat_list = get_pseudo_features(self.task_specific_batch_norm, training_phase, global_feat, domains)
        else :
            fake_feat_list = None

        return global_feat, bn_feat, cls_outputs, fake_feat_list ,dis
        '''
        if self.training:

            cls_score = self.classifier(bn_feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:

            return global_feat
        '''

    def freeze_layer(self,layer=None):
        # self.bottleneck.requires_grad_(False)
        # self.bottleneck.eval()
        self.base.freeze_layer(layer=layer)




__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

__factory_T_type_dual = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID_dual ,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID_dual ,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID_dual ,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID_dual 
}
def make_model(arg, num_class, camera_num, view_num,cfg,pretrain=True):
    if arg.MODEL == 'transformer':
        model = build_transformer(num_class, camera_num, view_num, cfg,arg, __factory_T_type_dual)
        if pretrain and cfg.MODEL.PRETRAIN_CHOICE=='imagenet':

            model.base.load_param(cfg.MODEL.PRETRAIN_PATH)

        print('===========building transformer===========')   
            
    else:
        model = Backbone(1, 'BN', False, False, Bottleneck, num_class, [3, 4, 6, 3])
        print('===========building ResNet===========')
        if pretrain:

            # cached_file = '/mnt/data/xukunlun/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
            # state_dict = torch.load(cached_file)
            # model.base.load_state_dict(state_dict, strict=False)

            import torchvision
            res_base = torchvision.models.resnet50(pretrained=True)
            res_base_dict = res_base.state_dict()

            state_dict = model.base.state_dict()

            for k, v in res_base_dict.items():
                if k in state_dict:
                    if v.shape == state_dict[k].shape:
                        state_dict[k] = v
                    else:
                        print('param {} of shape {} does not match loaded shape {}'.format(k, v.shape,
                                                                                           state_dict[k].shape))
                else:
                    print('param {} in pre-trained model does not exist in this model.base'.format(k))

            model.base.load_state_dict(state_dict, strict=True)
    return model
