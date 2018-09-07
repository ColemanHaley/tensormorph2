#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr, radial_basis
from tpr import *
from radial_basis import GaussianPool
from writer import Writer
from recorder import Recorder

# encapsulates main hierarchical attention logic of reading / writing
# attention distributions are computed from gradient (scalar) indices 
# that are updated over processing steps
# hierarchical structure of attention:
# - morpheme (stem vs. affix)
# - ordinal position within morpheme (0, 1, ..., nrole-1)
class Combiner(nn.Module):
    def __init__(self):
        super(Combiner, self).__init__()
        self.morph_attender = GaussianPool(2)
        self.posn_attender  = GaussianPool(tpr.nrole)
        self.writer = Writer()
        self.recorder = Recorder()


    def forward(self, stem, affix, copy, pivot, unpivot, max_len):
        nbatch = stem.shape[0]
        morph_attender = self.morph_attender
        posn_attender  = self.posn_attender
        writer = self.writer
        self.recorder.init()
        writer.init(nbatch)

        # initialize soft indices (all zeros)
        a  = torch.zeros(nbatch, 1, requires_grad=True) # morph (0.0=>stem, 1.0=>affix)
        b0 = torch.zeros(nbatch, 1, requires_grad=True) # position within stem
        b1 = torch.zeros(nbatch, 1, requires_grad=True) # position within affix
        c  = torch.zeros(nbatch, 1, requires_grad=True) # position within output

        for i in range(max_len):
            # map soft indices to attention distributions
            alpha = morph_attender(a)
            beta0 = posn_attender(b0)
            beta1 = posn_attender(b1)
            omega = posn_attender(c)

            # get copy, pivot, unpivot probabilities at current stem and affix positions
            theta       = alpha[:,0].unsqueeze(1)
            theta_stem  = dot_batch(pivot, beta0)
            theta_affix = dot_batch(unpivot, beta1)
            delta       = dot_batch(copy, beta0)

            # update tpr of output
            writer(stem, affix, alpha, beta0, beta1, omega, delta)

            if tpr.record:
                self.recorder.update_recording({
                    'morph_indx':a,
                    'stem_indx':b0,
                    'affix_indx':b1,
                    'output_indx':c,
                    'pivot_prob':theta_stem,
                    'unpivot_prob':theta_affix
                    })

            # update stem/affix selection and position within each morph
            # - switch morph at (un)pivot points, else stay
            a  = a + theta * theta_stem - (1.0 - theta) * theta_affix
            # - convex combos of advance within each morpheme and stay
            b0 = (1.0 - theta) * b0 + theta * (b0 + 1.0)
            b1 = theta * b1 + (1.0 - theta) * (b1 + 1.0)
            # - advance within output only if have copied from stem or affix
            c  = c + theta * delta + (1.0 - theta)
            # xxx reset affix position to 0 after unpivot (allowing for multiple affixation)
            #reset_affix = (1.0-theta)*theta2
            #x_affix = reset_affix*0.0 + (1.0-reset_affix)*x_affix

        output = writer.normalize()
        return output