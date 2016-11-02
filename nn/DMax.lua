--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
local DMax, parent = torch.class('nn.DMax', 'nn.Module')

function DMax:__init(dimension, windowSize, gpu)
    parent.__init(self)
    dimension = dimension or 1
    self.dimension = dimension
    self.windowSize = windowSize
    self.gpu = gpu
    self.max_modules = {}
    self.gradInput = {torch.Tensor()}
end

function DMax:updateOutput(inputs)
    local input, sizes = unpack(inputs)
    self.output:resize(sizes:size(1), input:size(2))
    local start_idx = 1
    for i = 1, sizes:size(1) do
        local max_module = self.max_modules[i]
        if max_module == nil then
            if self.gpu then
                self.max_modules[i] = nn.Max(self.dimension):cuda()
            else
                self.max_modules[i] = nn.Max(self.dimension)
            end
            max_module = self.max_modules[i]
        end
        self.output[i] = max_module:forward(input[{{start_idx, start_idx+sizes[i]-self.windowSize}}])
        start_idx = start_idx + sizes[i]
    end
    return self.output
end

function DMax:updateGradInput(inputs, gradOutput)
    local input, sizes = unpack(inputs)
    self.gradInput[1]:resizeAs(input):zero()
    self.gradInput[2] = self.gradInput[2] ~= nil and self.gradInput[2]:resizeAs(sizes):zero() or sizes.new():resizeAs(sizes):zero()
    local start_idx = 1
    for i = 1, sizes:size(1) do
        local max_module = self.max_modules[i]
        assert(max_module ~= nil)
        self.gradInput[1][{{start_idx, start_idx+sizes[i]-self.windowSize}}] = max_module:backward(input[{{start_idx, start_idx+sizes[i]-self.windowSize}}], gradOutput[i])
        start_idx = start_idx + sizes[i]

    end

    return self.gradInput
end
