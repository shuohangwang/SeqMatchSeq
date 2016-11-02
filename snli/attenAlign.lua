--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local attenAlign = torch.class('seqmatchseq.attenAlign')

function attenAlign:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.002
    self.emb_lr        = config.emb_lr        or 0
    self.emb_partial   = config.emb_partial   or true
    self.batch_size    = config.batch_size    or 25
    self.reg           = config.reg           or 0
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'snli'
    self.numWords      = config.numWords
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adamax'
    self.num_classes   = config.num_classes   or 3

    self.best_score    = 0

    self.emb_vecs = Embedding(self.numWords, self.emb_dim)
    self.emb_vecs.weight = tr:loadVacab2Emb(self.task):float()
    if self.emb_partial then
        self.emb_vecs.unUpdateVocab = tr:loadUnUpdateVocab(self.task)
    end

    self.dropoutl = nn.Dropout(self.dropoutP)
    self.dropoutr = nn.Dropout(self.dropoutP)



    self.optim_state = { learningRate = self.learning_rate }
    self.criterion = nn.ClassNLLCriterion()

    local lstm_config = {in_dim = self.emb_dim, mem_dim = self.mem_dim}


    local wwatten_config = {att_dim = self.att_dim, mem_dim = self.mem_dim }

     -- LSTM
    self.llstm = seqmatchseq.LSTM(lstm_config)
    self.rlstm = seqmatchseq.LSTM(lstm_config)
    --attention model
    self.att_module = seqmatchseq.LSTMwwatten(wwatten_config)
    --softmax
    self.soft_module = nn.Sequential():add(nn.Linear(self.mem_dim, self.num_classes)):add(nn.LogSoftMax())

    local modules = nn.Sequential():add(self.llstm):add(self.att_module):add(self.soft_module)
    self.params, self.grad_params = modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    share_params(self.rlstm, self.llstm)
end

function attenAlign:train(dataset)
    self.dropoutl:training()
    self.dropoutr:training()
    local indices = torch.randperm(dataset.size)
    local zeros = torch.zeros(self.mem_dim)
    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
            self.gradWeight = {}
            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]
                local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]

                local linputs_emb = self.emb_vecs:forward(lsent)
                local rinputs_emb = self.emb_vecs:forward(rsent)

                local linputs = self.dropoutl:forward(linputs_emb)
                local rinputs = self.dropoutr:forward(rinputs_emb)

                self.llstm:forward(linputs)
                self.rlstm:forward(rinputs)
                local lHinputs = self.llstm.hOutput
                local rHinputs = self.rlstm.hOutput

                local softInput = self.att_module:forward({lHinputs, rHinputs})
                local output = self.soft_module:forward(softInput[2])

                local loss = loss + self.criterion:forward(output, dataset.labels[idx])


                -- backpropagate
                local soft_grad = self.criterion:backward(output, dataset.labels[idx])
                local att_grad = self.soft_module:backward(softInput[2], soft_grad)

                local rep_grad = self:atten_backward(rsent, {lHinputs, rHinputs}, att_grad)

                local linputs_grad = self.llstm:backward(linputs, rep_grad[1])
                local rinputs_grad = self.rlstm:backward(rinputs, rep_grad[2])

                if self.emb_lr ~= 0 then
                    linputs_emb_grad = self.dropoutl:backward(linputs_emb, linputs_grad)
                    rinputs_emb_grad = self.dropoutr:backward(rinputs_emb, rinputs_grad)
                    self.emb_vecs:backward(lsent, linputs_emb_grad)
                    self.emb_vecs:backward(rsent, rinputs_emb_grad)
                end

            end

            loss = loss / batch_size
            self.grad_params:div(batch_size)

            -- regularization
            if self.reg ~= 0 then
                loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
                self.grad_params:add(self.reg, self.params)
            end

            return loss, self.grad_params
        end
        optim[self.grad](feval, self.params, self.optim_state)
		if self.emb_lr ~= 0 then
        	self.emb_vecs:updateParameters(self.emb_lr)
		end
    end
    xlua.progress(dataset.size, dataset.size)
end

function attenAlign:atten_backward(rsent, inputs, att_grad)
    local grad = torch.zeros(rsent:nElement(), self.mem_dim)
    grad[rsent:nElement()] = att_grad
    local rep_grad = self.att_module:backward(inputs, grad)
    return rep_grad
end

function attenAlign:predict(lsent, rsent)

    local linputs_emb = self.emb_vecs:forward(lsent)
    local rinputs_emb = self.emb_vecs:forward(rsent)

    local linputs = self.dropoutl:forward(linputs_emb)
    local rinputs = self.dropoutr:forward(rinputs_emb)


    self.llstm:forward(linputs)
    self.rlstm:forward(rinputs)
    local lHinputs = self.llstm.hOutput
    local rHinputs = self.rlstm.hOutput

    local softInput = self.att_module:forward({lHinputs, rHinputs})
    local output = self.soft_module:forward(softInput[2])

    self.att_module:forget()
    self.llstm:forget()
    self.rlstm:forget()

    local _,idx = torch.max(output,1)
    return idx
end

function attenAlign:predict_dataset(dataset)
    self.dropoutl:evaluate()
    self.dropoutr:evaluate()
    local predictions = torch.Tensor(dataset.size)
    local accuracy = 0
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
        predictions[i] = self:predict(lsent, rsent)
        if predictions[i] == dataset.labels[i] then
            accuracy = accuracy + 1
        end
    end
    return accuracy / dataset.size
end

function attenAlign:save(path, config, result, epoch)
    assert(string.sub(path,-1,-1)=='/')
    local paraPath     = path .. config.task .. config.expIdx
    local paraBestPath = path .. config.task .. config.expIdx .. '_best'
    local recPath      = path .. config.task .. config.expIdx ..'Record.txt'

    local file = io.open(recPath, 'a')
    if epoch == 1 then
        for name, val in pairs(config) do
            file:write(name .. '\t' .. tostring(val) ..'\n')
        end
    end

    file:write(config.task..': '..epoch..': ')
    for _, val in pairs(result) do
        print(val)
        file:write(val .. ', ')
    end
    file:write('\n')

    file:close()
    if result[1] > self.best_score then
        self.best_score  = result[1]
        self.best_params:copy(self.params)
        torch.save(paraBestPath, {params = self.params,config = config})
    end
    torch.save(paraPath, {params = self.params, config = config})
end

function attenAlign:load(path)
    local state = torch.load(path)
    self:__init(state.config)
    self.params:copy(state.params)
end
