from __future__ import absolute_import
from collections import OrderedDict
import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs,training_phase=None):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()
        NeedAux=False
        if len(inputs)%2>0:# 输入非整数，双卡测试将出错）
            NeedAux=True
            B,D,W,H=inputs.shape            
            auxiliary=torch.zeros((1,D,W,H)).to(inputs.device)
            inputs=torch.cat((inputs,auxiliary), dim=0)

        outputs = model(inputs,training_phase=training_phase)
        if NeedAux:
            outputs=outputs[:-1]
        outputs = outputs.data.cpu()
        return outputs


def extract_cnn_feature_two_model(model_new,model_old, inputs,training_phase=None):
    model_new.eval()
    model_old.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()
        outputs1 ,dis1 = model_new(inputs,training_phase=training_phase)
        outputs2 ,dis2 = model_old(inputs,training_phase=training_phase)
        if dis1.shape[1] == 4 :
            dis = torch.cat([dis1,dis2[:,:3]],dim=1)
            index = torch.argmin(dis,dim=1)
            print('num of choosing msmt model: ',sum(index==3))
            outputs = torch.where((index%4==3)[:,None],outputs1,outputs2)
            outputs = outputs.data.cpu()
        #print('outputs2',outputs2.shape,dis1.shape)
        else :
            outputs = torch.where(dis1[:,0]<dis2[:,0],outputs1,outputs2)
            print(sum(dis1[:,0]<dis2[:,0]))
            #print((dis1[:,0]<dis2[:,0]).reshape(-1))
            #print((dis1[:,0]-dis2[:,0]).reshape(-1))
            #print(outputs.shape)
            outputs = outputs.data.cpu()
        return outputs