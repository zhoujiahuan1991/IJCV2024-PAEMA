import pickle
import torch
import transformers
import copy
# import torch_scatter as ts
import torch.nn as nn
transformers.logging.set_verbosity(50)


class PromptPool(nn.Module):
  def __init__(self):
    super().__init__()
    self.total = None  #pool size per layer. if dual prompt : entire pool size = task num
    self.new = None #한 층에 새로 추가되는 key개수
    self.pnum = None #equal to 'expert prompt length' in dual prompt paper
    self.pdim = None #prompt dimension
    self.kdim = None #key dimension
    self.key_list = None #key를 저장하는 list
    self.prompt_list = None #prompt를 저장하는 list
    self.layer = None #prompt pool의 층 수
    #self.taskID_dict = {} #TIL일 때 task id가 주어지면 바로 prompt정보를 줄 수 있게 하는 dictionary
    
  # ! support uniform init (jax default => U(0, 0.01))
  def initPool(self,layer,total,pnum,pdim,kdim,embedding_layer=None,init_type='default'):
    self.layer = layer
    self.total = total #한 층에 해당하는 pool당 key 개수
    self.new = 0
    self.pnum = pnum
    self.pdim = pdim
    self.kdim = kdim
    self.init_type = init_type

    if embedding_layer != None: # initialize key token with word embedding
      print('------TODO: device------------')
      exit(0)

      
    else:
      if self.init_type == 'default':
        self.key_list = nn.Parameter(torch.randn(total,kdim))
      elif self.init_type == 'unif':
        self.key_list = nn.Parameter(0.01*torch.rand(total,kdim))
      else:
        raise ValueError('not supported init type')
      

    if embedding_layer != None: # initialize prompt token with word embedding
      embedding_layer
      self.prompt_list = []
      for i in range(self.layer):
        layer_pool = []

        for j in range(total):
          words = torch.randint(low=500,high=10000,size=(1,pnum))
          prompts = embedding_layer(words).squeeze().clone().detach()
          layer_pool.append(prompts.requires_grad_(True))
        self.prompt_list.append(layer_pool)
    
    else:
      if self.layer<=0:
        print("not using prompt-e, escaping prompt_list initialing...")
        return
      if self.init_type == 'default':
        self.prompt_list = nn.Parameter(torch.randn(self.layer,total,pnum,pdim))
      elif self.init_type == 'unif':
        self.prompt_list = nn.Parameter(0.01*torch.rand(self.layer,total,pnum,pdim))
      else:
        raise ValueError('not supported init type')
      '''
      self.prompt_list = []
      for i in range(self.layer):
        if self.init_type == 'default':
          self.prompt_list.append([torch.randn((pnum,pdim),requires_grad=True) for j in range(total)])
        elif self.init_type == 'unif':  # ! same with l2p initialization
          self.prompt_list.append([torch.tensor(0.01*torch.rand((pnum,pdim)),requires_grad=True) for j in range(total)])
        else:
          raise ValueError('not supported init type')
      '''
    #self.taskID_dict[len(self.taskID_dict.keys())] = self.total


