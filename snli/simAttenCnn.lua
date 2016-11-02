--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local simAttenCnn = torch.class('seqmatchseq.simAttenCnn')

function simAttenCnn:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.num_layers    = config.num_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.lstmModel     = config.lstmModel     or 'lstm' -- {lstm, bilstm}
    self.sim_nhidden   = config.sim_nhidden   or 50
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'paraphrase'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adamax'
    self.visualize     = config.visualize     or false
    self.directions    = config.directions    or 1
    self.num_classes   = config.num_classes   or 3
    self.sim_type      = config.sim_type      or 'sub'

    self.best_score    = 0

    self.lemb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.remb_vecs = nn.LookupTable(self.numWords, self.emb_dim)
    self.lemb_vecs.weight:copy(tr:loadVacab2Emb(self.task):float())

    self.dropoutl = nn.Dropout(self.dropoutP)
    self.dropoutr = nn.Dropout(self.dropoutP)

    self.llstm = seqmatchseq.LSTM({in_dim = self.emb_dim, mem_dim = self.mem_dim})
    self.rlstm = seqmatchseq.LSTM({in_dim = self.emb_dim, mem_dim = self.mem_dim})

    self.optim_state = { learningRate = self.learning_rate }

    self.criterion = nn.ClassNLLCriterion()

    self.latt_module = seqmatchseq.CNNwwSimatten({mem_dim = self.mem_dim, sim_type = self.sim_type})

    self.soft_module = nn.Sequential():add(nn.Linear(self.mem_dim*5, self.num_classes)):add(nn.LogSoftMax())

    local modules = nn.Parallel()
        --:add(self.lemb_vecs)
        :add(self.llstm)
        :add(self.latt_module)
        :add(self.soft_module)
    self.params, self.grad_params = modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    share_params(self.remb_vecs, self.lemb_vecs)
    share_params(self.rlstm, self.llstm)
end

function simAttenCnn:new_proj_module()
    local input = nn.Identity()()
    local i = nn.Sigmoid()(nn.Linear(self.emb_dim, self.mem_dim)(input))
    local u = nn.Tanh()(nn.Linear(self.emb_dim, self.mem_dim)(input))
    local output = nn.CMulTable(){i, u}
    local module = nn.gModule({input}, {output})
    if self.proj_module_master then
        share_params(module, self.proj_module_master)
    end
    return module
end


function simAttenCnn:train(dataset)
    self.dropoutl:training()
    self.dropoutr:training()
    self.soft_module:training()
    local indices = torch.randperm(dataset.size)
    local zeros = torch.zeros(self.mem_dim)

    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]
                local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
                local linputs = self.lemb_vecs:forward(lsent)
                local rinputs = self.remb_vecs:forward(rsent)
                linputs = self.dropoutl:forward(linputs)
                rinputs = self.dropoutr:forward(rinputs)

                local lHinputs = self.llstm:forward(linputs)
                local rHinputs = self.rlstm:forward(rinputs)
                lHinputs = self.llstm.hOutput
                rHinputs = self.rlstm.hOutput

                if lsent:size(1) == 1 then lHinputs:resize(1, self.mem_dim) end
                if rsent:size(1) == 1 then rHinputs:resize(1, self.mem_dim) end

                local lsimInput = self.latt_module:forward({lHinputs, rHinputs})
                local rsimInput, output
                output = self.soft_module:forward(lsimInput)


                local example_loss = self.criterion:forward(output, dataset.labels[idx])
                loss = loss + example_loss
                local soft_grad = self.criterion:backward(output, dataset.labels[idx])
                local att_grad = self.soft_module:backward(lsimInput, soft_grad)
                local lrep_grad = self.latt_module:backward({lHinputs, rHinputs}, att_grad)
                if lsent:size(1) == 1 then lrep_grad[1]:resize(self.mem_dim) end
                if rsent:size(1) == 1 then lrep_grad[2]:resize(self.mem_dim) end
                local llstm_grad = self.llstm:backward(linputs, lrep_grad[1])
                local rlstm_grad = self.rlstm:backward(rinputs, lrep_grad[2])
                --self.lemb_vecs:backward(lsent, llstm_grad)
                --self.remb_vecs:backward(rsent, rlstm_grad)
            end
            loss = loss / batch_size
            self.grad_params:div(batch_size)
            return loss, self.grad_params
        end
        optim[self.grad](feval, self.params, self.optim_state)
    end
    xlua.progress(dataset.size, dataset.size)
end


function simAttenCnn:predict(lsent, rsent)
    self.dropoutl:evaluate()
    self.dropoutr:evaluate()
    self.soft_module:evaluate()
    local linputs = self.lemb_vecs:forward(lsent)
    local rinputs = self.remb_vecs:forward(rsent)
    linputs = self.dropoutl:forward(linputs)
    rinputs = self.dropoutr:forward(rinputs)
    local lHinputs = self.llstm:forward(linputs)
    local rHinputs = self.rlstm:forward(rinputs)
    lHinputs = self.llstm.hOutput
    rHinputs = self.rlstm.hOutput
    if lsent:size(1) == 1 then lHinputs:resize(1, self.mem_dim) end
    if rsent:size(1) == 1 then rHinputs:resize(1, self.mem_dim) end

    local lsimInput = self.latt_module:forward({lHinputs, rHinputs})
    local output = self.soft_module:forward(lsimInput)

    self.llstm:forget()
    self.rlstm:forget()

    local _,idx = torch.max(output,1)
    return idx
end

function simAttenCnn:predict_dataset(dataset)
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

function simAttenCnn:save(path, config, result, epoch)
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

function simAttenCnn:load(path)
    local state = torch.load(path)
    if self.visualize then
        state.config.visualize = true
    end
    self:__init(state.config)
    self.params:copy(state.params)
end
