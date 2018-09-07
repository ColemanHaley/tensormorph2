#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *
from scanner import BiScanner, BiLSTMScanner
from stem_modifier import StemModifier
from thunker import Thunker
from combiner import Combiner
from recorder import Recorder

class Affixer(nn.Module):
    def __init__(self, node='root'):
        super(Affixer, self).__init__()
        self.scanner        = BiLSTMScanner(hidden_size = 1)
        self.pivoter        = BiScanner(morpho_size = tpr.dmorph+2, nfeature = 5, bias = 0.0)
        self.stem_modifier  = StemModifier()
        if node=='root':
            self.reduplicator = Affixer('reduplicant')
            self.unpivoter  = BiScanner(morpho_size = tpr.dmorph+2, nfeature = 5, bias = 0.0)
            self.redup      = Parameter(torch.zeros(1)) # xxx need to modulate by morph
        self.affix_thunker  = Thunker()
        self.combiner       = Combiner()
        self.node           = node
        self.recorder       = Recorder()


    # map tpr of stem to tpr of stem+affix
    def forward(self, stem, morph, max_len):
        nbatch  = stem.shape[0]
        scan    = torch.zeros((nbatch,2)) # self.scanner(stem)
        morpho  = torch.cat([morph, scan], 1)

        copy    = self.stem_modifier(stem, morpho)
        pivot   = self.pivoter(stem, morpho)
        affix, unpivot = self.get_affix(stem, morpho, max_len)

        output  = self.combiner(stem, affix, copy, pivot, unpivot, max_len)

        if tpr.record:
            self.recorder.set_recording({
                'stem_tpr':stem,
                'affix_tpr':affix,
                'copy':copy,
                'pivot':pivot,
                'unpivot':unpivot,
                'output_tpr':output
            })

        return output, affix, (pivot, copy, unpivot)


    def get_affix(self, stem, morpho, max_len):
        if self.node=='root':
            # reduplicative affix
            affix_redup = self.reduplicator(stem, morpho)
            pivot_redup = self.unpivoter(affix0, morpho)
            # non-reduplicative affix
            affix_fixed, pivot_fixed = self.affix_thunker(morpho)
            # convex combination of two affixes
            redup = torch.zeros() # sigmoid(self.redup)
            affix = redup * affix_redup + (1.0 - redup) * affix_fixed
            pivot = redup * pivot_redup + (1.0 - redup) * affix_fixed
        else:
            # enforce non-reduplicative affix
            affix, pivot = self.affix_thunker(morpho)

        return affix, pivot


    def init(self):
        print 'Affixer.init() does nothing'