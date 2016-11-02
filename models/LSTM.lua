--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
local LSTM, parent = torch.class('seqmatchseq.LSTM', 'nn.Module')

function LSTM:__init(config)
    parent.__init(self)

    self.in_dim = config.in_dim
    self.mem_dim = config.mem_dim or 150

    if config.output_gate ~= nil then self.output_gate = config.output_gate else self.output_gate = true end

    self.master_cell = self:new_cell()
    self.depth = 0
    self.hOutput = torch.Tensor()
    self.cells = {}

    self.initial_values = {torch.zeros(self.mem_dim), torch.zeros(self.mem_dim)}
    self.gradInput = { torch.zeros(self.in_dim),torch.zeros(self.mem_dim),torch.zeros(self.mem_dim) }
end

function LSTM:new_cell()
    local input = nn.Identity()()
    local c_p = nn.Identity()()
    local h_p = nn.Identity()()

    local new_gate = function()
        return nn.CAddTable(){
            nn.Linear(self.in_dim, self.mem_dim)(input),
            nn.Linear(self.mem_dim, self.mem_dim)(h_p)
        }
    end

    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local u = nn.Tanh()(new_gate())
    local o

    if self.output_gate then o = nn.Sigmoid()(new_gate()) end

    local c = nn.CAddTable(){ nn.CMulTable(){f, c_p}, nn.CMulTable(){i, u} }

    local h
    if self.output_gate then
        h = nn.CMulTable(){o, nn.Tanh()(c)}
    else
        h = nn.Tanh()(c)
    end

    local cell = nn.gModule({input, c_p, h_p}, {c, h})

    if self.master_cell then
        share_params(cell, self.master_cell)
    end
    return cell
end

function LSTM:forward(inputs, reverse)
    local size = inputs:size(1)
    self.hOutput:resize(size, self.mem_dim)
    for t = 1, size do
        local idx = reverse and size-t+1 or t
        local input = inputs[idx]

        self.depth = self.depth + 1
        local cell = self.cells[self.depth]
        if cell == nil then
            cell = self:new_cell()
            self.cells[self.depth] = cell
        end

        local prev_output = self.depth > 1 and self.cells[self.depth - 1].output or self.initial_values

        local outputs = cell:forward({input, prev_output[1], prev_output[2]})
        local c, h = unpack(outputs)

        self.output = h
        self.hOutput[idx] = h
    end

    return self.output
end

function LSTM:backward(inputs, grad_outputs, reverse)
    local size = inputs:size(1)
    assert(self.depth ~= 0)

    local input_grads = torch.Tensor(inputs:size())
    for t = size, 1, -1 do
        local idx = reverse and size-t+1 or t
        local input = inputs[idx]
        local grad_output = grad_outputs[idx]

        local cell = self.cells[self.depth]
        local grads = {self.gradInput[2], self.gradInput[3]}
        grads[2]:add(grad_output)

        local prev_output = self.depth > 1 and self.cells[self.depth - 1].output or self.initial_values

        self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)

        input_grads[idx] = self.gradInput[1]

        self.depth = self.depth - 1
    end

    self:forget()

    return input_grads
end

function LSTM:share(lstm, ...)
    assert( self.in_dim == lstm.in_dim )
    assert( self.mem_dim == lstm.mem_dim )
    share_params(self.master_cell, lstm.master_cell, ...)
end

function LSTM:zeroGradParameters()
    self.master_cell:zeroGradParameters()
end

function LSTM:parameters()
    return self.master_cell:parameters()
end

function LSTM:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end
