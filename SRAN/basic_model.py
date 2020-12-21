import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.model

    def load_model(self, path, epoch): 
        state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch))
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch):
        torch.save(self.state_dict(), path+'{}_epoch_{}.pth'.format(self.name, epoch))       

    def compute_loss(self, output, target, meta_target):
        pass


