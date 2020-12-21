import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from basic_model import BasicModel

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()    
    def forward(self, x):
        return x

class SRAN(BasicModel):
    def __init__(self, args):
        super(SRAN, self).__init__(args)

        self.conv1 = models.resnet18(pretrained=False)
        self.conv1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.fc = identity()

        self.conv2 = models.resnet18(pretrained=False)
        self.conv2.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2.fc = identity()

        self.conv3 = models.resnet18(pretrained=False)
        self.conv3.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv3.fc = identity()

        self.dataset = args.dataset

        if self.dataset == 'I-RAVEN':
            self.meta_target_length = 9
        elif self.dataset == 'PGM':
            self.meta_target_length = 12

        self.img_size = args.img_size

        gate_function_1 = [nn.Linear(3*512, 3*512), nn.ReLU(), nn.Linear(3*512, 512), nn.ReLU()]
        gate_function_2 = [nn.Linear(4*512, 4*512), nn.ReLU(), nn.Linear(4*512, 512), nn.ReLU()]
        gate_function_3 = [nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024,512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 512+self.meta_target_length)]

        self.h1 = nn.Sequential(*gate_function_1)
        self.h2 = nn.Sequential(*gate_function_2)
        self.h3 = nn.Sequential(*gate_function_3)
     
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_beta = args.meta_beta
        self.row_to_column_idx = [0,3,6,1,4,7,2,5]

    def compute_loss(self, output, target, meta_target):
        pred, meta_target_pred, = output[0], output[1]
        target_loss = F.cross_entropy(pred, target)     
        BCE_loss =  torch.nn.BCEWithLogitsLoss()
        meta_target_loss = BCE_loss(meta_target_pred, meta_target)
        #print (target_loss.item(), meta_target_loss.item())
        loss = target_loss + self.meta_beta*meta_target_loss
        return loss

    def get_pair(self, row1, row2):
        x = torch.cat((row1, row2), dim = 1)
        y = torch.cat((row2, row1), dim = 1)
        z = torch.stack((x, y), dim = 1)
        return z

    def get_row_rules(self, x, panel_features):

        row3 = x[:,6:8,:,:].unsqueeze(1)
        #B×1×2×224×224
        row3_candidates = torch.cat((row3.repeat(1,8,1,1,1), x[:,8:16,:,:].unsqueeze(2)), dim = 2)
        #B×8×3×224×224
        row_all = torch.cat((x[:,0:3,:,:].unsqueeze(1), x[:,3:6,:,:].unsqueeze(1), row3_candidates), dim = 1)
        #B×10×3×224×224
        intra_row_relations = self.conv2(row_all.view(-1,3,self.img_size,self.img_size))
        #(10B)×512
        

        choice_rows =  torch.cat((x[:,6:8,:,:].unsqueeze(1).repeat(1,8,1,1,1), x[:,8:16,:,:].unsqueeze(2)), dim = 2)
        #B×8×3×224×224
        conv_row_list = [self.get_pair(x[:,0:3,:,:], x[:,3:6,:,:])] + [self.get_pair(x[:,0:3,:,:], choice_rows[:,i,:,:,:]) for i in range(8)] + \
                        [self.get_pair(x[:,3:6,:,:], choice_rows[:,i,:,:,:]) for i in range(8)]
        conv_rows = torch.stack(conv_row_list, dim = 1)
        #B×17×2×6×224×224
        inter_row_relations = self.conv3(conv_rows.view(-1,6,self.img_size,self.img_size))
        #(34B)×512
        inter_row_relations = torch.sum(inter_row_relations.view(-1,17,2,512), dim = 2)
        #B×17×512  


        row3_12features = panel_features[:,6:8,:].unsqueeze(1).repeat(1,8,1,1)
        #B×8×2×512
        candidate_features = panel_features[:,8:16,:].unsqueeze(2)
        #B×8×1×512
        row3_features = torch.cat((row3_12features,candidate_features), dim = 2)
        #B×8×3×512
        row_features = [panel_features[:,0:3,:].unsqueeze(1), panel_features[:,3:6,:].unsqueeze(1), row3_features]
        row_features = torch.cat(row_features, dim = 1)
        #B×10×3×512
        row_relations = self.h1(row_features.view(-1,1536))
        #(10B)×512
        row_relations = torch.cat((row_relations, intra_row_relations), dim = 1)
        #(10B)×1024
        row_relations = row_relations.view(-1,10,1024)
        #B×10×1024
        row_list = [self.get_pair(row_relations[:,0,:], row_relations[:,1,:])] + [self.get_pair(row_relations[:,0,:], row_relations[:,i,:]) for i in range(2,10)] + \
                   [self.get_pair(row_relations[:,1,:], row_relations[:,i,:]) for i in range(2,10)]
        row_relations = torch.stack(row_list, dim = 1)
        #B×17×2×2048
        rules = self.h2(row_relations.view(-1,2048))
        #(34B)×512
        rules = torch.sum(rules.view(-1,17,2,512), dim = 2)
        #B×17×512        
        rules = torch.cat((rules, inter_row_relations), dim = 2)
        #B×17×1024
        rules = self.h3(rules)
        #B×17×(512+L)        

        return rules[:,:,:512], rules[:,:,512:]

    def row_to_column(self, x, panel_features): 
        context_image = x[:,self.row_to_column_idx]
        image = torch.cat((context_image, x[:,8:,:,:]), dim = 1)
        context_features = panel_features[:,self.row_to_column_idx]
        features = torch.cat((context_features, panel_features[:,8:,:]), dim = 1)
        return image, features

    def forward(self, x):
        B = x.size(0)
        panel_features = self.conv1(x.view(-1,1,self.img_size,self.img_size))
        #(16B)×512
        panel_features = panel_features.view(-1,16,512)
        #B×16×512

        row_output = self.get_row_rules(x, panel_features) 
        row_rules, meta_target_row_pred = row_output[0], row_output[1]
        #B×17×512
        #B×17×L
   
        if self.dataset == 'I-RAVEN':            
            column_rules = torch.zeros(B,17,512).cuda()
            meta_target_column_pred = torch.zeros(B,17,self.meta_target_length).cuda()
        else:      
            x_c, panel_features_c = self.row_to_column(x, panel_features)
            column_output = self.get_row_rules(x_c, panel_features_c)
            column_rules, meta_target_column_pred = column_output[0], column_output[1]
            #B×17×512
            #B×17×L        

        rules = torch.cat((row_rules, column_rules), dim = 2)
        #B×17×1024
        meta_target_pred = meta_target_row_pred[:,0,:] + meta_target_column_pred[:,0,:]
        #B×L

        dominant_rule = rules[:,0,:].unsqueeze(1)
        pseudo_rules = rules[:,1:,:]
        similarity = torch.bmm(dominant_rule, torch.transpose(pseudo_rules, 1, 2)).squeeze(1)
        #B×16
        similarity = similarity[:,:8] + similarity[:,8:]
        #B×8 

        return similarity, meta_target_pred
        