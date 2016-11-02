--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local pointNet, parent = torch.class('seqmatchseq.pointNet', 'nn.Module')

function pointNet:__init(config)
    parent.__init(self)
    self.in_dim   = config.in_dim   or config.mem_dim
    self.mem_dim  = config.mem_dim  or 150
    self.dropoutP = config.dropoutP or 0
    self.master_pt = self:new_pt()
    self.depth = 0
    self.hOutput = torch.Tensor()
    self.pts = {}

    self.initial_values = {torch.zeros(self.mem_dim), torch.zeros(self.mem_dim)}
    self.gradInput = {
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim)
    }

    self.join_module = nn.JoinTable(2)
    self.join_drop_module = nn.Sequential():add(nn.JoinTable(2)):add(nn.Dropout(self.dropoutP))
end

function pointNet:new_pt()
    local H_con = nn.Identity()()
    local c_p = nn.Identity()()
    local h_p = nn.Identity()()

    local H_pad = nn.Padding(1,1)(H_con)
    local M_H_pad = nn.Linear(self.in_dim, self.mem_dim)(H_pad)
    local M_h_p = nn.Linear(self.mem_dim, self.mem_dim)(h_p)
    local M_h = nn.Tanh()(nn.CAddRepTable(){M_H_pad, M_h_p})

    local M = nn.Linear(self.mem_dim, 1)(M_h)
    local alpha = nn.SoftMax()( nn.View(-1)(M) )
    local Y =  nn.MV(true){H_pad, alpha}
    local alpha_log = nn.LogSoftMax()(nn.View(-1)(M))

    local new_gate = function()
        return nn.CAddTable(){
        nn.Linear(self.in_dim, self.mem_dim)(Y),
        nn.Linear(self.mem_dim, self.mem_dim)(h_p)
        }
    end

    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local update = nn.Tanh()(new_gate())


    local c = nn.CAddTable(){
      nn.CMulTable(){f, c_p},
      nn.CMulTable(){i, update}
    }
    local o = nn.Sigmoid()(new_gate())
    local h = nn.CMulTable(){o, nn.Tanh()(c)}

    local pt = nn.gModule({H_con, c_p, h_p}, {c, h, alpha_log})


    if self.master_pt then
        share_params(pt, self.master_pt)
    end
    return pt
end


function pointNet:forward(inputs)
    local H, H_b, labels = unpack(inputs)
    self.H_con = self.join_drop_module:forward{H, H_b}
    self.hOutput:resize(labels:size(1), self.H_con:size(1)+1)
    local loss_avg = 0
    for t = 1, labels:size(1) do
        self.depth = self.depth + 1
        local pt = self.pts[self.depth]
        if pt == nil then
            pt = self:new_pt()
            self.pts[self.depth] = pt
        end
        local prev_output = t > 1 and self.pts[self.depth - 1].output or self.initial_values
        local pt_out = pt:forward({self.H_con, prev_output[1], prev_output[2]})

        assert(pt_out[3]:size(1)==self.H_con:size(1)+1)

        self.hOutput[t] = pt_out[3]
    end
    return self.hOutput

end

function pointNet:backward(inputs, grad)
    local H, H_b, labels = unpack(inputs)
    local input_grads = torch.zeros(self.H_con:size())
    local params, grad_params = self.master_pt:parameters()

    for t = labels:size(1), 1, -1 do
        assert(self.depth ~= 0)
        local pt = self.pts[self.depth]
        local prev_output = t > 1 and self.pts[self.depth - 1].output or self.initial_values

        self.gradInput = pt:backward({self.H_con, prev_output[1], prev_output[2]},
                                     {self.gradInput[2], self.gradInput[3], grad[t]})
        input_grads:add(self.gradInput[1])
        self.depth = self.depth - 1
    end
    local H_grad = self.join_drop_module:backward({H, H_b}, input_grads)
    self:forget()
    return H_grad
end
function pointNet:predict(inputs, lens)
    local H, H_b = unpack(inputs)
    local labels = {}
    self.H_con = self.join_module:forward{H, H_b}
    local loss_avg = 0
    while #labels < 30 do
        self.depth = self.depth + 1
        local pt = self.pts[self.depth]
        if pt == nil then
            pt = self:new_pt()
            self.pts[self.depth] = pt
        end
        local prev_output = self.depth > 1 and self.pts[self.depth - 1].output or self.initial_values
        local pt_out = pt:forward({self.H_con, prev_output[1], prev_output[2]})
        local _,idx = torch.max(pt_out[3],1)
        if idx[1] == self.H_con:size(1)+1 then
            if lens then
                _,idx = torch.max(pt_out[3][{{1,-2}}],1)
            else
                break
            end
        end
        labels[#labels+1] = idx[1]
        if lens == #labels then break end
    end
    return labels

end

function pointNet:predict_search(inputs, lens)
    local H, H_b = unpack(inputs)
    local labels = {}
    self.H_con = self.join_module:forward{H, H_b}
    local loss_avg = 0
    local labels_tensor = torch.zeros(2, self.H_con:size(1))
    for i = 1, 2 do
        self.depth = self.depth + 1
        local pt = self.pts[self.depth]
        if pt == nil then
            pt = self:new_pt()
            self.pts[self.depth] = pt
        end
        local prev_output = self.depth > 1 and self.pts[self.depth - 1].output or self.initial_values
        local pt_out = pt:forward({self.H_con, prev_output[1], prev_output[2]})
        labels_tensor[i]:copy( pt_out[3][{{1,-2}}] )
    end
    local max_score = -999999
    for i = 1, labels_tensor:size(2) do
        for j = 0, 15 do
            if i + j > labels_tensor:size(2) then break end
            local score = labels_tensor[1][i] + labels_tensor[2][i+j]

            if score > max_score then
                max_score = score
                labels = {i, i+j}
            end
        end
    end
    return labels
end

function pointNet:predict_prob(inputs, lens)
    local H, H_b = unpack(inputs)
    local labels = {}
    self.H_con = self.join_module:forward{H, H_b}
    local loss_avg = 0
    local labels_tensor = torch.zeros(2, self.H_con:size(1))
    for i = 1, 2 do
        self.depth = self.depth + 1
        local pt = self.pts[self.depth]
        if pt == nil then
            pt = self:new_pt()
            self.pts[self.depth] = pt
        end
        local prev_output = self.depth > 1 and self.pts[self.depth - 1].output or self.initial_values
        local pt_out = pt:forward({self.H_con, prev_output[1], prev_output[2]})
        labels_tensor[i]:copy( pt_out[3][{{1,-2}}] )
    end

    return labels_tensor
end

function pointNet:share(pointNet, ...)
    if self.in_dim ~= pointNet.in_dim then error("pointNet input dimension mismatch") end
    if self.mem_dim ~= pointNet.mem_dim then error("pointNet memory dimension mismatch") end
    share_params(self.master_pt, pointNet.master_pt, ...)
end

function pointNet:zeroGradParameters()
    self.master_pt:zeroGradParameters()
end

function pointNet:parameters()
    return self.master_pt:parameters()
end

function pointNet:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end
