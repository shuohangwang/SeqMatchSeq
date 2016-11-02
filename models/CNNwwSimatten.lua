--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local CNNwwSimatten, parent = torch.class('seqmatchseq.CNNwwSimatten', 'nn.Module')

function CNNwwSimatten:__init(config)
    parent.__init(self)
    self.mem_dim   = config.mem_dim
    self.att_dim   = config.att_dim  or self.mem_dim
    self.fih_dim   = config.fih_dim  or self.mem_dim
    self.sim_type  = config.sim_type
    self.cov_dim   = self.sim_type == 'cos' and 2 or self.mem_dim
    self.master_ww = self:new_ww()
end

function CNNwwSimatten:new_ww()
    local linput, rinput = nn.Identity()(), nn.Identity()()
    --padding
    local lPad = nn.Padding(1,1)(linput)
    local M_l = nn.Linear(self.mem_dim, self.mem_dim)(lPad)
    local M_r = nn.MM(false, true){M_l, rinput}

    local alpha = nn.SoftMax()( nn.Transpose({1,2})(M_r) )

    local Yl =  nn.MM(){alpha, lPad}

    local sim_out
    if self.sim_type == 'submul' then
        local sub = nn.Power(2)(nn.CSubTable(){Yl, rinput})
        local mul = nn.CMulTable(){Yl, rinput}
        sim_out = nn.ReLU()(nn.Linear(2*self.mem_dim, self.mem_dim)( nn.JoinTable(2){sub, mul}))
    elseif self.sim_type == 'sub' then
        sim_out = nn.Power(2)(nn.CSubTable(){Yl, rinput})
    elseif self.sim_type == 'mul' then
        sim_out = nn.CMulTable(){Yl, rinput}
    elseif self.sim_type == 'weightsub' then
        sim_out = nn.Power(2)(nn.CSubTable(){nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(Yl)), nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(rinput))})
    elseif self.sim_type == 'weightmul' then
        sim_out = nn.CMulTable(){nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(Yl)), nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(rinput))}
    elseif self.sim_type == 'bilinear' then
        sim_out = nn.ReLU()(nn.Bilinear(self.mem_dim,self.mem_dim, self.mem_dim)({Yl, rinput}))
    elseif self.sim_type == 'concate' then
        sim_out = nn.ReLU()(nn.Linear(2*self.mem_dim, self.mem_dim)( nn.JoinTable(2){Yl, rinput}))
    elseif self.sim_type == 'cos' then
        local cos = nn.View(-1,1)(nn.CosineDistance(){Yl, rinput})
        local dis = nn.View(-1,1)(nn.PairwiseDistance(2){Yl, rinput})
    	sim_out= nn.JoinTable(2){cos, dis}
    else
        print(self.sim_type)
        assert(false)
    end

    local cnnIn = nn.Padding(1,2)(nn.Padding(1,-2)(sim_out))

    local conv1 = nn.ReLU()(nn.TemporalConvolution(self.cov_dim, self.fih_dim, 1)(cnnIn))
    local conv2 = nn.ReLU()(nn.TemporalConvolution(self.cov_dim, self.fih_dim, 2)(cnnIn))
    local conv3 = nn.ReLU()(nn.TemporalConvolution(self.cov_dim, self.fih_dim, 3)(cnnIn))
    local conv4 = nn.ReLU()(nn.TemporalConvolution(self.cov_dim, self.fih_dim, 4)(cnnIn))
    local conv5 = nn.ReLU()(nn.TemporalConvolution(self.cov_dim, self.fih_dim, 5)(cnnIn))

    local pool1 = nn.Max(1)(conv1)
    local pool2 = nn.Max(1)(conv2)
    local pool3 = nn.Max(1)(conv3)
    local pool4 = nn.Max(1)(conv4)
    local pool5 = nn.Max(1)(conv5)

    local output = nn.JoinTable(1){pool1, pool2, pool3, pool4, pool5}

    local ww = nn.gModule({linput, rinput}, {output})

    if self.master_ww then
        share_params(ww, self.master_ww)
    end

    return ww

end

function CNNwwSimatten:forward(inputs)
    self.output = self.master_ww:forward(inputs)
    return self.output
end

function CNNwwSimatten:backward(inputs, grad_outputs)
    self.gradInput = self.master_ww:backward(inputs, grad_outputs)

    return self.gradInput
end



function CNNwwSimatten:share(CNNwwSimatten, ...)
    if self.att_dim ~= CNNwwSimatten.att_dim then error("CNNwwSimatten attention dimension mismatch") end
    if self.mem_dim ~= CNNwwSimatten.mem_dim then error("CNNwwSimatten memory dimension mismatch") end
    share_params(self.master_ww, CNNwwSimatten.master_ww, ...)
end

function CNNwwSimatten:zeroGradParameters()
    self.master_ww:zeroGradParameters()
end

function CNNwwSimatten:parameters()
    return self.master_ww:parameters()
end
