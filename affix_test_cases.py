#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re, string, sys
sys.path.append('/home/coleman/Documents/wilsonlab/tensormorphy')

import tpr, trainer
from seq_embedder import SeqEmbedder, string2sep
from morph_embedder import MorphEmbedder
from tpr import *

maindir = '~/Documents/wilsonlab/tensormorphy/'

def import_data(data_select):
    print('importing data')
    dat, held_in_stems, held_out_stems, syms = None, None, None, None

    # select data set, initialize global variables
    if data_select == 'english_ing':
        datfile = '~/Dropbox/AlbrightHayes2003/RuleBasedLearnerEnglishFiles/CELEXverbs.txt'
        dat = pd.read_table(datfile, sep='\t', header=-1)
        dat.columns = ['freq', 'stem', 'output', 'type']
        dat['stem'] = [str(x).lower() for x in dat.stem]
        dat = dat[(dat.type=='reg')]
        dat = dat[(~dat.stem.str.contains('-'))]
        dat['output'] = [x+'ing' for x in dat.stem] # xxx pseudo-ing
        #dat['output'] = ['over'+x for x in dat.stem] # xxx pseudo-over
        dat = dat[(dat.stem.str.len() <= 6)] # (nrole-2))]
        dat = dat.reset_index()
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [string2sep(x) for x in dat.output]
        dat['morph'] = '???'
        vowels = ['i', 'e', 'a', 'o', 'u']
        morph_embedder = MorphEmbedder.get_embedder(None, None)
        #from string import ascii_lowercase
        #syms  = [x for x in ascii_lowercase]

    if data_select == 'english_ness':  # English -ness suffixation
        datfile = maindir + 'english/english_ness.csv'
        dat = pd.read_table(datfile, sep=',')
        dat['stem'] = dat.stem.str.lower()
        dat['output_orig'] = dat.output
        dat['output'] = [x+'ness' for x in dat.stem]
        dat['output'] = dat.output.str.lower()
        dat = dat[(dat.output.str.len() < 11)]
        dat.reset_index()
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [string2sep(x) for x in dat.output]
        dat['morph'] = '???'
        vowels = ['i', 'e', 'a', 'o', 'u']
        morph_embedder = MorphEmbedder.get_embedder(None, None)

    if data_select == 'english_un':  # English un- prefixation
        datfile = maindir + 'english_morphology/english_un.csv'
        dat = pd.read_table(datfile, sep=',')
        dat['stem'] = dat.stem.str.lower()
        dat['output_orig'] = dat.output
        dat['output'] = ['un'+x for x in dat.stem]
        dat['output'] = dat.output.str.lower()
        dat = dat[(dat.output.str.len() < 20)]
        dat.reset_index()
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [string2sep(x) for x in dat.output]
        vowels = ['i', 'e', 'a', 'o', 'u']
        morph_embedder = MorphEmbedder.get_embedder(None, None)

    if data_select == 'english_shm':  # English shm- reduplication
        datfile = maindir + 'english/english_shm_reduplication.txt'
        dat = pd.read_table(datfile, sep=',', header=-1, comment='#')
        dat.columns     = ['output']
        dat['stem']     = [re.sub(' -.*', '', x).lower() for x in dat['output']]
        dat['output']   = [re.sub('.*- ', '', x).lower() for x in dat['output']]
        dat = dat[(~dat.stem.str.contains(' ')) & (dat.output.str.len()<13)]
        dat.reset_index()
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [string2sep(x) for x in dat.output]
        dat['morph'] = '???'
        vowels = ['i', 'e', 'a', 'o', 'u']
        syms = [x for x in string.ascii_lowercase]
        morph_embedder = MorphEmbedder.get_embedder(None, None)

    if data_select == 'chamorro_um':  # Chamorro -um- infixation
        datfile = maindir + 'chamorro_infixation/chamorro_um_clean_all.csv'
        dat = pd.read_table(datfile, sep=',', usecols=[0,1])
        dat = dat[(dat.output.str.len() < 20)]
        dat.reset_index()
        if 0:   # reverse Chamorro
            dat['stem'] = [x[::-1] for x in dat.stem]
            dat['output'] = [x[::-1] for x in dat.output]
        dat['stem']     = [string2sep(x) for x in dat.stem]
        dat['output']   = [string2sep(x) for x in dat.output]
        dat['morph']    = '???'
        substitutions = [(u'P',u'ʔ'), (u'c h', u'ts'), (u'y', u'dz'), (u'n g', u'ŋ')]
        for (x,y) in substitutions:
            dat['stem']   = [re.sub(x,y,stem) for stem in dat.stem]
            dat['output'] = [re.sub(x,y,output) for output in dat.output]
        held_in_stems   = [string2sep(x) for x in ['brabu', 'planta']]
        held_out_stems  = [string2sep(x) for x in ['tristi',]]
        vowels          = ['i', 'e', 'a', 'o', 'u']
        morph_embedder = MorphEmbedder.get_embedder(None, None)

        dat_v = dat[(dat.stem.str.match('^[ieaou]'))]
        dat_c = dat[(dat.stem.str.match('^[^ieaou] [ieaou]'))]
        dat_cc = dat[(dat.stem.str.match('^[^ieaou] [^ieaou]'))]
        print(len(dat_v), len(dat_c), len(dat_cc))
        #print dat_v
        #print dat_cc
        # compensate for small number of V-initial stems
        #dat = pd.concat([dat, dat_v, dat_v, dat_v])
        # restrict to c-initial stems
        #dat = dat_c # pd.concat([dat_c, dat_cc])
        #dat.reset_index()

    if data_select == 'hungarian_dat':  # Hungarian dative
        datfile = maindir + 'hungarian_suffixation/hungarian_dat_sg.csv'
        dat = pd.read_table(datfile, sep=',', encoding='utf-8')
        dat['stem'] = dat.stem.str.lower()
        dat['output'] = dat.output.str.lower()
        dat = dat[~(dat.output.str.contains(' '))]
        dat = dat[~(dat.output.str.contains('[.-]'))]
        dat = dat[(dat.output.str.contains('(nak|nek)$'))]
        dat = dat[(dat.output.str.len() < 10)]
        dat.reset_index()
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [string2sep(x) for x in dat.output]
        dat['morph'] = '???'
        vowels = [u'a', u'e', u'i', u'o', u'u', u'á', u'é', u'í', u'ó', u'ö', u'ú', u'ü', u'ő', u'ű']
        morph_embedder = MorphEmbedder.get_embedder(None, None)

    if data_select == 'hebrew_paal':  # Hebrew pa'al conjugation
        datfile = maindir + 'hebrew/verbs_eran_tomer/hebrew_paal_real.csv'
        dat = pd.read_table(datfile, sep=',', encoding='utf-8')
        dat = dat[(dat.morph=='IMPERATIVE+SECOND+F+SINGULAR+COMPLETE')]
        dat.reset_index()
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [string2sep(x) for x in dat.output]
        substitutions = [(u't s', u'ts'), (u'ħ', u'Χ'), (u'ʕ', u'ʔ'),\
        (u'ʕ', u'ʔ'), (u'q', u'k')]
        for (x,y) in substitutions:
            dat['stem'] = [re.sub(x, y, stem) for stem in dat.stem]
            dat['output'] = [re.sub(x, y, output) for output in dat.output]
        vowels = [u'i', u'e', u'a', u'o', u'u', u'ə']
        #morph_embedder = MorphEmbedder.get_embedder('hebrew', None)
        morph_embedder = MorphEmbedder.get_embedder(None, None)

    if data_select == 'hindi_nouns':
        datfile = maindir +'hindi/nouns/all.txt'
        dat = pd.read_table(datfile, sep=',', encoding='utf-8')
        dat = dat[(dat.output.str.len() < 20)]
        dat.reset_index()
        dat['stem'] = [re.sub('^.*  ', '', x) for x in dat.stem]
        dat['output'] = [re.sub('^.*  ', '', x) for x in dat.output]
        vowels = [u'a', u'A', u'i',	u'I', u'u', u'U', u'e', u'E', u'o', u'O']
        morph_embedder = MorphEmbedder.get_embedder('hindi', None)

    if data_select == 'conll':
        lang, quantity = 'finnish', ['low', 'medium', 'high'][0]
        print(lang)
        datfile = maindir +'conll-sigmorphon2018/task1/all/'+ lang +'-train-'+ quantity
        dat1 = pd.read_table(datfile, sep='\t', encoding='utf-8')
        dat1.columns = ['stem', 'output', 'morph']
        dat1['type'] = 'train'

        datfile = maindir +'conll-sigmorphon2018/task1/all/'+ lang +'-dev'
        dat2 = pd.read_table(datfile, sep='\t', encoding='utf-8')
        dat2.columns = ['stem', 'output', 'morph']
        dat2['type'] = 'dev'

        dat = dat1.append(dat2)
        dat = dat[(dat.output.str.len() < 15)]
        dat.reset_index()
        dat['stem'] = [re.sub(u' ', u'_', x) for x in dat.stem]
        dat['stem'] = [string2sep(x) for x in dat.stem]
        dat['output'] = [re.sub(u' ', u'_', x) for x in dat.output]
        dat['output'] = [string2sep(x) for x in dat.output]
        morph_embedder = MorphEmbedder.get_embedder(None, dat)
        #morph_embedder.reduce_dimension(dat)
        vowels = [u'i', u'e', u'a', u'o', u'u', u'í', u'é', u'á', u'ó', u'ú', u'ü']

    print(dat.head())
    print('number of stem-output pairs:', len(dat))
    #print dat.output

    # collect symbols from data if not already specified
    syms = set([x for y in dat.output for x in y.split(' ')])\
        if syms is None else set(syms)
    syms = sorted([x for x in syms])
    rando = [False, True][0]
    seq_embedder = SeqEmbedder(syms, vowels=vowels, nrole=30, random_fillers=rando, random_roles=rando)
    #print 'symbols: ', ' '.join(syms)

    # remove held_in and held_out before random split
    if held_in_stems is not None:
        held_in = dat[(dat.stem.isin(held_in_stems))]
        dat      = dat[~(dat.stem.isin(held_in_stems))]
        dat.reset_index()
    if held_out_stems is not None:
        held_out = dat[(dat.stem.isin(held_out_stems))]
        dat      = dat[~(dat.stem.isin(held_out_stems))]
        dat.reset_index()

    if data_select != 'conll':
        train, test = trainer.make_split(dat)
    else:
        train = dat[(dat['type']=='train')]
        test  = dat[(dat['type']=='dev')]

    # place held-out forms at the top of test
    if held_in_stems is not None:
        train = pd.concat([held_in, test])
        train.reset_index()
    if held_out_stems is not None:
        test = pd.concat([held_out, test])
        test.reset_index()
    print(train.head())
    print(test.head())

    return seq_embedder, morph_embedder, train, test
