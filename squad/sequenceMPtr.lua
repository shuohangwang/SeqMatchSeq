--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
local sequenceMPtr = torch.class('seqmatchseq.sequenceMPtr')

function sequenceMPtr:__init(config)
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
    self.visualize     = false
    self.emb_lr        = config.emb_lr        or 0.001
    self.emb_partial   = config.emb_partial   or true
    self.best_res      = 0

    self.emb_vecs = Embedding(self.numWords, self.emb_dim)
    self.emb_vecs.weight = tr:loadVacab2Emb(self.task):float()
    if self.emb_partial then
        self.emb_vecs.unUpdateVocab = tr:loadUnUpdateVocab(self.task)
    end
    self.dropoutl = nn.Dropout(self.dropoutP)
    self.dropoutr = nn.Dropout(self.dropoutP)

    self.optim_state = { learningRate = self.learning_rate }

    self.criterion = nn.ClassNLLCriterion()

    self.llstm = seqmatchseq.LSTM({in_dim = self.emb_dim, mem_dim = self.mem_dim, output_gate = false})
    self.rlstm = seqmatchseq.LSTM({in_dim = self.emb_dim, mem_dim = self.mem_dim, output_gate = false})

    self.att_module = seqmatchseq.LSTMwwatten({mem_dim = self.mem_dim, att_dim = self.att_dim})
    self.att_module_b = seqmatchseq.LSTMwwatten({mem_dim = self.mem_dim, att_dim = self.att_dim})

    self.point_module = seqmatchseq.pointNet({in_dim = 2*self.mem_dim, mem_dim = self.mem_dim, dropoutP = self.dropoutP / 2})
    local modules = nn.Parallel()
        :add(self.llstm)
        :add(self.att_module)
        :add(self.point_module)
    self.params, self.grad_params = modules:getParameters()

    share_params(self.rlstm, self.llstm)
    share_params(self.att_module_b, self.att_module)
end


function sequenceMPtr:train(dataset)
    self.llstm:training()
    self.rlstm:training()
    self.dropoutl:training()
    self.dropoutr:training()
    dataset.size = #dataset
    local indices = torch.randperm(dataset.size)
    local zeros = torch.zeros(self.mem_dim)
    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
	        self.emb_vecs.gradWeight = {}
            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]
                local lsent, rsent, labels = unpack(dataset[idx])

                local linputs_emb = self.emb_vecs:forward(lsent)
                local rinputs_emb = self.emb_vecs:forward(rsent)

                local linputs = self.dropoutl:forward(linputs_emb)
                local rinputs = self.dropoutr:forward(rinputs_emb)


                local rep_inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
                local lHinputs = self.llstm.hOutput
                local rHinputs = self.rlstm.hOutput


                self.att_module:forward({rHinputs, lHinputs})
                self.att_module_b:forward({rHinputs, lHinputs}, true)

                local point_out = self.point_module:forward({self.att_module.hOutput, self.att_module_b.hOutput, labels})

                local example_loss = self.criterion:forward(point_out, labels)

                loss = loss + example_loss

                -- backpropagate
                local crt_grad = self.criterion:backward(point_out, labels)
                local pt_grad = self.point_module:backward({self.att_module.hOutput, self.att_module_b.hOutput, labels}, crt_grad)

                local att_grad = self.att_module:backward({rHinputs, lHinputs}, pt_grad[1])
                local att_b_grad = self.att_module_b:backward({rHinputs, lHinputs}, pt_grad[2], true)
                local linputs_grad = self.llstm:backward(linputs, att_grad[2]+att_b_grad[2])
                local rinputs_grad = self.rlstm:backward(rinputs, att_grad[1]+att_b_grad[1])
                if self.emb_lr ~= 0 then
                    linputs_emb_grad = self.dropoutl:backward(linputs_emb, linputs_grad)
                    rinputs_emb_grad = self.dropoutr:backward(rinputs_emb, rinputs_grad)
                    self.emb_vecs:backward(lsent, linputs_emb_grad)
                    self.emb_vecs:backward(rsent, rinputs_emb_grad)
                end
            end

            loss = loss / batch_size
            self.grad_params:div(batch_size)
            print('loss: '..loss)
            print(self.grad_params:mean()..'; '..self.emb_vecs.weight:mean())
            return loss, self.grad_params
        end

        optim[self.grad](feval, self.params, self.optim_state)
        if self.emb_lr ~= 0 then
            self.emb_vecs:updateParameters(self.emb_lr)
	    end
        break
    end
    xlua.progress(dataset.size, dataset.size)
end

function sequenceMPtr:trainDt(dataset)
    self.llstm:training()
    self.rlstm:training()
    self.dropoutl:training()
    self.dropoutr:training()
    dataset.size = #dataset
    self.grad_params:zero()
    self.emb_vecs.gradWeight = {}
    local loss = 0
    for idx = 1, dataset.size do
        local lsent, rsent, labels = unpack(dataset[idx])

        local linputs_emb = self.emb_vecs:forward(lsent)
        local rinputs_emb = self.emb_vecs:forward(rsent)

        local linputs = self.dropoutl:forward(linputs_emb)
        local rinputs = self.dropoutr:forward(rinputs_emb)


        local rep_inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
        local lHinputs = self.llstm.hOutput
        local rHinputs = self.rlstm.hOutput


        self.att_module:forward({rHinputs, lHinputs})
        self.att_module_b:forward({rHinputs, lHinputs}, true)

        local point_out = self.point_module:forward({self.att_module.hOutput, self.att_module_b.hOutput, labels})

        local example_loss = self.criterion:forward(point_out, labels)

        loss = loss + example_loss

        -- backpropagate
        local crt_grad = self.criterion:backward(point_out, labels)
        local pt_grad = self.point_module:backward({self.att_module.hOutput, self.att_module_b.hOutput, labels}, crt_grad)

        local att_grad = self.att_module:backward({rHinputs, lHinputs}, pt_grad[1])
        local att_b_grad = self.att_module_b:backward({rHinputs, lHinputs}, pt_grad[2], true)
        local linputs_grad = self.llstm:backward(linputs, att_grad[2]+att_b_grad[2])
        local rinputs_grad = self.rlstm:backward(rinputs, att_grad[1]+att_b_grad[1])
        if self.emb_lr ~= 0 then
            linputs_emb_grad = self.dropoutl:backward(linputs_emb, linputs_grad)
            rinputs_emb_grad = self.dropoutr:backward(rinputs_emb, rinputs_grad)
            self.emb_vecs:backward(lsent, linputs_emb_grad)
            self.emb_vecs:backward(rsent, rinputs_emb_grad)
        end
    end
    loss = loss / dataset.size
    self.grad_params:div(dataset.size)
end

function sequenceMPtr:predict(lsent, rsent)

    local linputs_emb = self.emb_vecs:forward(lsent)
    local rinputs_emb = self.emb_vecs:forward(rsent)

    local linputs = self.dropoutl:forward(linputs_emb)
    local rinputs = self.dropoutr:forward(rinputs_emb)


    local rep_inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
    local lHinputs = self.llstm.hOutput
    local rHinputs = self.rlstm.hOutput


    self.att_module:forward({rHinputs, lHinputs})
    self.att_module_b:forward({rHinputs, lHinputs}, true)

    local point_out = self.point_module:predict({self.att_module.hOutput, self.att_module_b.hOutput})


    self.point_module:forget()
    self.att_module:forget()
    self.att_module_b:forget()

    self.llstm:forget()
    self.rlstm:forget()

    return point_out
end


function sequenceMPtr:predict_dataset(dataset)
    self.llstm:evaluate()
    self.rlstm:evaluate()
    self.dropoutl:evaluate()
    self.dropoutr:evaluate()
    local ivocab = tr:loadiVocab(self.task)
    local fileL = io.open('../trainedmodel/evaluation/squad/dev_output.txt', "w")
    dataset.size = #dataset
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local lsent, rsent = unpack(dataset[i])
        local predictions = self:predict(lsent, rsent)
        local pred_str = {}
        for j = 1, #predictions do
            pred_str[j] = ivocab[lsent[predictions[j] ] ]
            fileL:write(pred_str[j])
            if j ~= #predictions then fileL:write(' ') end
        end
        fileL:write('\n')
    end
    sys.execute('python ../trainedmodel/evaluation/squad/txt2js.py ../data/squad/dev-v1.1.json ../trainedmodel/evaluation/squad/dev_output.txt ../trainedmodel/evaluation/squad/prediction.json')
    local res = sys.execute('python ../trainedmodel/evaluation/squad/evaluate-v1.1.py ../data/squad/dev-v1.1.json ../trainedmodel/evaluation/squad/prediction.json')

    fileL:close()
    return res
end

function sequenceMPtr:predict_datasetDt(dataset)
    self.llstm:evaluate()
    self.rlstm:evaluate()
    self.dropoutl:evaluate()
    self.dropoutr:evaluate()
    local predictions = {}
    local ivocab = tr:loadiVocab(self.task)
    dataset.size = #dataset--/10
    for i = 1, dataset.size do
      local lsent, rsent = unpack(dataset[i])
      local pred = self:predict(lsent, rsent)
      local pred_str = ''
      for j = 1, #pred do
          pred_str = pred_str .. ivocab[lsent[pred[j] ] ]
          if j ~= #pred then pred_str = pred_str .. ' ' end
      end
      predictions[i] = pred_str .. '\n'
    end
    return predictions

end

function sequenceMPtr:save(path, config, result, epoch)
    assert(string.sub(path,-1,-1)=='/')
    local paraPath = path .. config.task .. config.expIdx
    local paraBestPath = path .. config.task .. config.expIdx .. '_best'
    local recPath = path .. config.task .. config.expIdx ..'Record.txt'

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

    torch.save(paraPath, {
        params = self.params,
        config = config,
        emb    = self.emb_lr ~= 0 and self.emb_vecs.weight or nil
    })

    local res = stringx.split(stringx.split(result[1], ',')[1],' ')[2]
    res = tonumber(res)
    if res > self.best_res then
        self.best_res = res
        torch.save(paraBestPath, {
            params = self.params,
            config = config,
            emb    = self.emb_lr ~= 0 and self.emb_vecs.weight or nil
        })
    end
end

function sequenceMPtr:load(path)
    local state = torch.load(path)
    if self.visualize then
        state.config.visualize = true
    end
    self:__init(state.config)
    self.params:copy(state.params)
    if state.emb ~= nil then
        self.emb_vecs.weight:copy(state.emb)
    end
end
