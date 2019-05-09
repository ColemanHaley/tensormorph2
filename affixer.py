#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *
from scanner import BiScanner, BiLSTMScanner
from stem_modifier import StemModifier
from thunker import Thunker
from combiner import Combiner

class Affixer(nn.Module):
    def __init__(self, node='root'):
        super(Affixer, self).__init__()
        self.scanner        = BiLSTMScanner(hidden_size = 1)
        self.pivoter        = BiScanner(morpho_size = tpr.dmorph+2, nfeature = 5, node = node+'-pivoter')
        self.stem_modifier  = StemModifier()
        self.analyze_stem   = nn.LSTM(tpr.nrole, 1, bidirectional=True)
        self.affix_attender = nn.Linear(31 * 2, 5)
        if node=='root':
            self.reduplicator = Affixer('reduplicant')
            self.unpivoter  = BiScanner(morpho_size = tpr.dmorph+2, nfeature = 5, node = node+'-unpivoter')
            self.redup      = Parameter(torch.zeros(1)) # xxx need to modulate by morph
        self.affix_thunker  = Thunker()
        self.combiner       = Combiner()
        self.node           = node


    # map tpr of stem to tpr of stem+affix
    def forward(self, stem, morph, max_len):
        nbatch  = stem.shape[0]
        scan    = torch.zeros((nbatch,2)) # self.scanner(stem)
        morpho  = torch.cat([morph, scan], 1)

        analyzed_stem = self.analyze_stem(stem)[0]
        affix_attn = torch.softmax(self.affix_attender(torch.cat((analyzed_stem[:, :, 0], analyzed_stem[:, :, 1]), dim=1)), dim=1)
        self.affix_attn_data = torch.softmax(self.affix_attender(torch.cat((analyzed_stem[:, :, 0], analyzed_stem[:, :, 1]), dim=1)), dim=1)

        copy_stem = self.stem_modifier(stem, morpho) if 1\
                    else torch.ones((nbatch, tpr.nrole))
        pivot   = self.pivoter(stem, morpho)
        affix, unpivot, copy_affix = self.get_affix(stem, morpho, max_len, affix_attn)
        output  = self.combiner(stem, affix, copy_stem, copy_affix, pivot, unpivot, max_len)

        if tpr.recorder is not None:
            tpr.recorder.set_values(self.node, {
                'stem_tpr':stem,
                'affix_tpr':affix,
                'copy_stem':copy_stem,
                'copy_affix':copy_affix,
                'pivot':pivot,
                'unpivot':unpivot,
                'output_tpr':output
            })

        return output, affix, (pivot, copy_stem, unpivot, copy_affix)


    def get_affix(self, stem, morpho, max_len, affix_attn):
        if self.node=='root':
            # reduplicative affix
            # xxx fixme
            #affix_redup = self.reduplicator(stem, morpho, max_len)
            #pivot_redup = self.unpivoter(affix0, morpho)
            # non-reduplicative affix

            affix_fixed, pivot_fixed, copy_fixed = self.affix_thunker(morpho)
            # convex combination of two affixes
            #redup = torch.zeros() # sigmoid(self.redup)
            #affix = redup * affix_redup + (1.0 - redup) * affix_fixed
            #pivot = redup * pivot_redup + (1.0 - redup) * affix_fixed
            #print(affix_attn)
            self.affixes = affix_fixed
            affix = torch.einsum('ba,ba...->b...', (affix_attn, affix_fixed))
            pivot = torch.einsum('ba,ba...->b...', (affix_attn, pivot_fixed))
            copy = torch.einsum('ba,ba...->b...', (affix_attn, copy_fixed))
        else:
            # enforce non-reduplicative affix
            affix, pivot, copy = self.affix_thunker(morpho)

            self.affixes = affix

            affix = torch.einsum('ba,ba...->b...', (affix_attn, affix))
            pivot = torch.einsum('ba,ba...->b...', (affix_attn, pivot))
            copy = torch.einsum('ba,ba...->b...', (affix_attn, copy))

        return affix, pivot, copy


    def init(self):
        print('Affixer.init() does nothing')
