--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
local CAddRepTable, parent = torch.class('nn.CAddRepTable', 'nn.Module')

function CAddRepTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CAddRepTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
       for j = 1, self.output:size(1) do
           self.output[j]:add(input[i])
       end
   end
   return self.output
end

function CAddRepTable:updateGradInput(input, gradOutput)
   for i=1,#input do

      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
      if i == 1 then
          self.gradInput[i]:copy(gradOutput)
      else
          self.gradInput[i]:copy(gradOutput:sum(1))
      end

   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
