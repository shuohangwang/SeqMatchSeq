--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local LSTMwwatten, parent = torch.class('seqmatchseq.LSTMwwatten', 'nn.Module')

function LSTMwwatten:__init(config)
    parent.__init(self)
    self.mem_dim       = config.mem_dim       or 150
    self.att_dim       = config.att_dim       or self.mem_dim
    self.in_dim       = config.in_dim       or self.mem_dim

    self.master_ww = self:new_ww()
    self.depth = 0
    self.wws = {}

    self.initial_values = {torch.zeros(self.mem_dim), torch.zeros(self.mem_dim)}
    self.gradInput = {
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim)
    }

    self.hOutput = torch.Tensor()
end

function LSTMwwatten:new_ww()

    local linput, rinput, c_p, m_p = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    --padding
    local lPad = nn.Padding(1,1)(linput)
    local M_l = nn.Linear(self.in_dim, self.att_dim)(lPad)

    local M_r = nn.Linear(self.in_dim, self.att_dim)(rinput)
    local M_a = nn.Linear(self.mem_dim, self.att_dim)(m_p)

    local M_ra =  nn.CAddTable(){M_r, M_a}
    local M = nn.Tanh()(nn.CAddRepTable(){M_l, M_ra})

    local wM = nn.Linear(self.att_dim, 1)(M)
    local alpha = nn.SoftMax()( nn.View(-1)(wM) )

    local Yl =  nn.MV(true){lPad, alpha}

    local new_gate = function()
        return nn.CAddTable(){
            nn.Linear(self.mem_dim, self.mem_dim)(m_p),
            nn.Linear(self.in_dim, self.mem_dim)(rinput),
            nn.Linear(self.in_dim, self.mem_dim)(Yl)
        }
    end

    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local u = nn.Tanh()(new_gate())
    local o = nn.Sigmoid()(new_gate())

    local c = nn.CAddTable(){
        nn.CMulTable(){f, c_p},
        nn.CMulTable(){i, u}
    }

    local m = nn.CMulTable(){o, nn.Tanh()(c)}


    local ww = nn.gModule({linput, rinput, c_p, m_p}, {c, m})

    if self.master_ww then
        share_params(ww, self.master_ww)
    end
    return ww
end

function LSTMwwatten:forward(inputs, reverse)
    local lHinputs, rHinputs = unpack(inputs)
    local size = rHinputs:size(1)
    self.hOutput:resize(size, self.mem_dim)
    for t = 1, size do
        local idx = reverse and size-t+1 or t
        self.depth = self.depth + 1
        local ww = self.wws[self.depth]
        if ww == nil then
            ww = self:new_ww()
            self.wws[self.depth] = ww
        end
        local prev_output = (self.depth > 1) and self.wws[self.depth - 1].output
                                             or self.initial_values

        local output = ww:forward({lHinputs, rHinputs[idx], unpack(prev_output)})
        self.hOutput[idx] = output[2]
        self.output = output
    end
    return self.output
end

function LSTMwwatten:backward(inputs, grad_outputs, reverse)
    local lHinputs, rHinputs = unpack(inputs)
    local size = rHinputs:size(1)
    local grad_lHinputs = torch.zeros(lHinputs:size())
    local grad_rHinputs = torch.zeros(rHinputs:size())
    assert( self.depth ~= 0 )

    for t = size, 1, -1 do
        local idx = reverse and size-t+1 or t
        local ww = self.wws[self.depth]
        local grad = {self.gradInput[3], self.gradInput[4]}
        grad[2]:add(grad_outputs[idx])

        local prev_output = (self.depth > 1) and self.wws[self.depth - 1].output
                                             or self.initial_values

        self.gradInput = ww:backward({lHinputs, rHinputs[idx], unpack(prev_output)}, grad)

        grad_lHinputs:add(self.gradInput[1])
        grad_rHinputs[idx] = self.gradInput[2]

        self.depth = self.depth - 1
    end
    self:forget()

    return {grad_lHinputs, grad_rHinputs}
end



function LSTMwwatten:share(LSTMwwatten)
    assert( self.att_dim == LSTMwwatten.att_dim )
    assert( self.mem_dim == LSTMwwatten.mem_dim )
    share_params(self.master_ww, LSTMwwatten.master_ww)
end

function LSTMwwatten:zeroGradParameters()
    self.master_ww:zeroGradParameters()
end

function LSTMwwatten:parameters()
    return self.master_ww:parameters()
end

function LSTMwwatten:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end
