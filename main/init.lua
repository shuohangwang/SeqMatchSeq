--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
require 'nngraph'
require 'optim'
require 'debug'
require 'nn'
require 'parallel'
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

seqmatchseq = {}
include '../util/loadFiles.lua'
include '../util/utils.lua'

include '../models/Embedding.lua'
include '../models/LSTM.lua'
include '../models/LSTMwwatten.lua'
include '../models/pointNet.lua'

include '../nn/CAddRepTable.lua'

include '../squad/sequenceMPtr.lua'
include '../squad/boundaryMPtr.lua'
opt = {}

opt.batch_size = 6
opt.num_processes = 5
opt.max_epochs = 30
opt.seed = 123
opt.reg = 0
opt.learning_rate = 0.002
opt.lr_decay = 0.95
opt.num_layers = 1
opt.m_layers = 2
opt.dropoutP = 0.4

opt.model = 'boundaryMPtr'
opt.task = 'squad'
opt.preEmb = 'glove'
opt.grad = 'adamax'
opt.expIdx = 0
opt.initialUNK = false
opt.emb_lr = 0
opt.emb_partial = true

opt.wvecDim = 300
opt.mem_dim = 150
opt.att_dim = 150

opt.log = 'log information'

opt.visualize = false
opt.numWords = 124164
torch.manualSeed(opt.seed)
