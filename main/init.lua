require 'nngraph'
require 'optim'
require 'debug'
require 'nn'
require 'parallel'
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

transition = {}
include '../util/loadFiles.lua'
include '../util/utils.lua'

include '../models/Embedding.lua'
include '../models/LSTM.lua'
include '../models/LSTMwwatten.lua'
include '../models/pointNet.lua'

include '../nn/CAddRepTable.lua'

include '../snli/attenAlign.lua'
include '../squad/pointMlstm.lua'
include '../squad/pointBEMlstm.lua'
include '../squad/bpointBEMlstm.lua'
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

opt.model = 'pointBEMlstm'
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
