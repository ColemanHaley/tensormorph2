#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *


# affix look-up table implemented with linear map from morphosyn + stem scan
class Thunker(nn.Module):
    def __init__(self, redup=False, root=True):
        super(Thunker, self).__init__()
        self.morph2affix =\
            nn.Linear(tpr.dmorph+2, tpr.dfill * tpr.drole * 5, bias=True)
        self.morph2unpivot =\
            nn.Linear(tpr.dmorph+2, tpr.nrole * 5, bias=True)
        self.morph2copy =\
            nn.Linear(tpr.dmorph+2, tpr.nrole * 5, bias=True)
        #self.morph2affix.bias.data.fill_(-2.5)
        #self.morph2unpivot.bias.data.fill_(2.5)

    def forward(self, morpho):
        nbatch = morpho.shape[0]
        # tpr of affix via binding matrix: affix tpr = F B_affix R^T
        #B_affix = self.morph2affix(morpho).view(nbatch, tpr.nfill, tpr.nrole)
        #B_affix = torch.exp(log_softmax(B_affix, dim=1)) # normalize within roles
        #affix = torch.bmm(torch.bmm(F, B_affix), Rt)
        # tpr of affix directly
        affix = self.morph2affix(morpho).view(nbatch, tpr.dfill, tpr.drole * 5)
        affix = sigmoid(affix) # restrict learned affix components to [0,1]
        #affix.data[0,0] = 1.0 # force affix to begin at 0th position
        #affix = tpr.seq_embedder.string2tpr('u m', False).unsqueeze(0).expand(nbatch, tpr.dfill, tpr.drole)   # xxx testing
        #affix = tanh(affix) # restrict learned affix components to [-1, +1]
        #affix  = tanh(PReLu(affix)) # restrict learned affix components to [0,1]
        #affix = bound_batch(affix)
        #affix = torch.zeros((nbatch, tpr.dfill, tpr.drole)) # xxx hack
        unpivot = self.morph2unpivot(morpho)
        unpivot = sigmoid(unpivot).view(nbatch, tpr.nrole * 5)
        copy = self.morph2copy(morpho)
        copy = sigmoid(copy).view(nbatch, tpr.nrole * 5)

        if tpr.discretize:
            affix = torch.round(affix)
            unpivot = torch.round(pivot)
            copy = torch.round(copy)

        return torch.stack(affix.chunk(5, dim=2), dim=1), torch.stack(unpivot.chunk(5, dim = 1),dim=1), torch.stack(copy.chunk(5, dim=1), dim=1)
