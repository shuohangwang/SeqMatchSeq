--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local wikiqaSimAttenCnn = torch.class('seqmatchseq.wikiqaSimAttenCnn')

function wikiqaSimAttenCnn:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.cov_dim       = config.cov_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.num_layers    = config.num_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.lstmModel     = config.lstmModel     or 'lstm' -- {lstm, bilstm}
    self.sim_nhidden   = config.sim_nhidden   or 50
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'wikiqa'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adamax'
    self.visualize     = false
    self.emb_lr        = config.emb_lr        or 0.001
    self.emb_partial   = config.emb_partial   or true
    self.sampleN       = config.sampleN       or 50
    self.sim_type      = config.sim_type      or 'concate'
    self.window_sizes  = {1,2,3,4,5}
    self.window_large  = self.window_sizes[#self.window_sizes]

    self.best_score    = 0
    
    self.emb_vecs = Embedding(self.numWords, self.emb_dim)
    self.emb_vecs.weight:copy( tr:loadVacab2Emb(self.task):float() )


    self.proj_module_master = self:new_proj_module()
    self.att_module_master = self:new_att_module()

    if self.sim_type == 'concate' then
        self.sim_sg_module = self:new_sim_con_module()
    elseif self.sim_type == 'sub' then
        self.sim_sg_module = self:new_sim_sub_module()
    elseif self.sim_type == 'mul' then
        self.sim_sg_module = self:new_sim_mul_module()
    elseif self.sim_type == 'weightsub' then
        self.sim_sg_module = self:new_sim_weightsub_module()
    elseif self.sim_type == 'weightmul' then
        self.sim_sg_module = self:new_sim_weightmul_module()
    elseif self.sim_type == 'bilinear' then
        self.sim_sg_module = self:new_sim_bilinear_module()
    elseif self.sim_type == 'submul' then
        self.sim_sg_module = self:new_sim_submul_module()
    elseif self.sim_type == 'cos' then
        self.cov_dim = 2
        self.sim_sg_module = self:new_sim_cos_module()
    else
        assert(false)
    end

    self.conv_module = self:new_conv_module()
    self.soft_module = nn.Sequential()
                       :add(nn.Linear(self.mem_dim, 1))
                       :add(nn.View(-1))
                       :add(nn.LogSoftMax())

    self.dropout_modules = {}
    self.proj_modules = {}
    self.att_modules = {}

    self.join_module = nn.JoinTable(1)

    self.optim_state = { learningRate = self.learning_rate }
    self.criterion = nn.DistKLDivCriterion()


    local modules = nn.Parallel()
        :add(self.proj_module_master)
        :add(self.att_module_master)
        :add(self.conv_module)
        :add(self.soft_module)
    self.params, self.grad_params = modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    print(self.params:size())
    for i = 1, 2 do
        self.proj_modules[i] = self:new_proj_module()
        self.dropout_modules[i] = nn.Dropout(self.dropoutP)
    end

end

function wikiqaSimAttenCnn:new_proj_module()
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

function wikiqaSimAttenCnn:new_att_module()
    local linput, rinput = nn.Identity()(), nn.Identity()()
    --padding
    local lPad = nn.Padding(1,1)(linput)
    --local M_l = nn.Linear(self.mem_dim, self.mem_dim)(lPad)

    local M_r = nn.MM(false, true){lPad, rinput}

    local alpha = nn.SoftMax()( nn.Transpose({1,2})(M_r) )

    local Yl =  nn.MM(){alpha, lPad}

    local att_module = nn.gModule({linput, rinput}, {Yl})

    if self.att_module_master then
        share_params(att_module, self.att_module_master)
    end

    return att_module
end
function wikiqaSimAttenCnn:new_conv_module()
    local input, sizes = nn.Identity()(), nn.Identity()()

    local conv = {}
    local pool = {}
    for i, window_size in pairs(self.window_sizes) do
        conv[i] = nn.ReLU()(nn.TemporalConvolution(self.cov_dim, self.mem_dim, window_size)(input))
        pool[i] = nn.DMax(1, window_size){conv[i], sizes}
    end
    local concate =  nn.JoinTable(2)(pool)
    local output = nn.Tanh()( nn.Linear(#self.window_sizes*self.mem_dim, self.mem_dim)(concate))

    local module = nn.gModule({input, sizes}, {output})
    return module
end


function wikiqaSimAttenCnn:new_sim_con_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.ReLU()(nn.Linear(2*self.mem_dim, self.mem_dim)(nn.JoinTable(2){inputq, inputa}))
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end


function wikiqaSimAttenCnn:new_sim_submul_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local qa_sub = nn.Power(2)(nn.CSubTable(){inputq, inputa})
    local qa_mul = nn.CMulTable(){inputq, inputa}
    local join = nn.JoinTable(2){qa_sub, qa_mul}
    local output = nn.ReLU()(nn.Linear(2*self.mem_dim, self.mem_dim)(join))

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function wikiqaSimAttenCnn:new_sim_bilinear_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.ReLU()(nn.Bilinear(self.mem_dim, self.mem_dim, self.mem_dim)({inputq, inputa}))
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function wikiqaSimAttenCnn:new_sim_sub_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.Power(2)(nn.CSubTable(){inputq, inputa})
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function wikiqaSimAttenCnn:new_sim_mul_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()

    local output = nn.CMulTable(){inputq, inputa}

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end
function wikiqaSimAttenCnn:new_sim_weightsub_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.Power(2)(nn.CSubTable(){nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputq)), nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputa))})
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function wikiqaSimAttenCnn:new_sim_weightmul_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()

    local output = nn.CMulTable(){nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputq)), nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputa))}

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function wikiqaSimAttenCnn:new_sim_cos_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local cos = nn.View(-1,1)(nn.CosineDistance(){inputq, inputa})
    local dis = nn.View(-1,1)(nn.PairwiseDistance(2){inputq, inputa})
    local output = nn.JoinTable(2){cos, dis}

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end


function wikiqaSimAttenCnn:train(dataset)
    for i = 1, 2 do
        self.proj_modules[i]:training()
        self.dropout_modules[i]:training()
    end
    self.conv_module:training()
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
                local data_raw = dataset[idx]
                local data_q = data_raw[1]
                local data_as = data_raw[2]
                local label = data_raw[3]
                local data_as_len = torch.IntTensor(#data_as)

                for k = 1, #data_as do
                    data_as_len[k] = data_as[k]:size(1)
                    if data_as_len[k] < self.window_large then
                        local as_tmp = torch.Tensor(self.window_large):fill(1)
                        as_tmp:sub(1,data_as_len[k]):copy(data_as[k])
                        data_as[k] = as_tmp
                        data_as_len[k] = self.window_large
                    end
                end

                local data_as_word = self.join_module:forward(data_as)
                local inputs_a_emb = self.emb_vecs:forward(data_as_word)
                local inputs_q_emb = self.emb_vecs:forward(data_q)

                inputs_a_emb = self.dropout_modules[1]:forward(inputs_a_emb)
                inputs_q_emb = self.dropout_modules[2]:forward(inputs_q_emb)

                local projs_a_emb = self.proj_modules[1]:forward(inputs_a_emb)
                local projs_q_emb = self.proj_modules[2]:forward(inputs_q_emb)
                if data_q:size(1) == 1 then projs_q_emb:resize(1, self.mem_dim) end

                local att_output = self.att_module_master:forward({projs_q_emb, projs_a_emb})

                local sim_output = self.sim_sg_module:forward({projs_a_emb, att_output})

                local conv_output = self.conv_module:forward({sim_output, data_as_len})
                local soft_output = self.soft_module:forward(conv_output)
                local example_loss = self.criterion:forward(soft_output, label)

                loss = loss + example_loss

                local crit_grad = self.criterion:backward(soft_output, label)
                local soft_grad = self.soft_module:backward(conv_output, crit_grad)


                local conv_grad = self.conv_module:backward({sim_output, data_as_len}, soft_grad)


                local sim_grad = self.sim_sg_module:backward({projs_a_emb, att_output}, conv_grad[1])

                local att_grad = self.att_module_master:backward({projs_q_emb, projs_a_emb}, sim_grad[2])

                local projs_a_emb = self.proj_modules[1]:backward(inputs_a_emb, att_grad[2]+sim_grad[1])
                local projs_q_emb = self.proj_modules[2]:backward(inputs_q_emb, att_grad[1])


            end
            loss = loss / batch_size
            self.grad_params:div(batch_size)
            return loss, self.grad_params
        end

        optim[self.grad](feval, self.params, self.optim_state)
        collectgarbage()

    end
    xlua.progress(dataset.size, dataset.size)
end


function wikiqaSimAttenCnn:predict(data_raw)
    local data_q = data_raw[1]
    local data_as = data_raw[2]
    local label = data_raw[3]
    local data_as_len = torch.IntTensor(#data_as)

    for k = 1, #data_as do
        data_as_len[k] = data_as[k]:size(1)
        if data_as_len[k] < self.window_large then
            local as_tmp = torch.Tensor(self.window_large):fill(1)
            as_tmp:sub(1,data_as_len[k]):copy(data_as[k])
            data_as[k] = as_tmp
            data_as_len[k] = self.window_large
        end
    end

    local data_as_word = self.join_module:forward(data_as)
    local inputs_a_emb = self.emb_vecs:forward(data_as_word)
    local inputs_q_emb = self.emb_vecs:forward(data_q)

    inputs_a_emb = self.dropout_modules[1]:forward(inputs_a_emb)
    inputs_q_emb = self.dropout_modules[2]:forward(inputs_q_emb)

    local projs_a_emb = self.proj_modules[1]:forward(inputs_a_emb)
    local projs_q_emb = self.proj_modules[2]:forward(inputs_q_emb)
    if data_q:size(1) == 1 then projs_q_emb:resize(1, self.mem_dim) end

    local att_output = self.att_module_master:forward({projs_q_emb, projs_a_emb})

    local sim_output = self.sim_sg_module:forward({projs_a_emb, att_output})

    local conv_output = self.conv_module:forward({sim_output, data_as_len})
    local soft_output = self.soft_module:forward(conv_output)
    local map = MAP(label,soft_output)
    local mrr = MRR(label,soft_output)
    return {map,mrr}
end

function wikiqaSimAttenCnn:predict_dataset(dataset)
    for i = 1, 2 do
        self.proj_modules[i]:evaluate()
        self.dropout_modules[i]:evaluate()
    end
    self.conv_module:evaluate()
    local res = {0,0}
    dataset.size = #dataset--/10
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local prediction = self:predict(dataset[i])
        res[1] = res[1] + prediction[1]
        res[2] = res[2] + prediction[2]
    end
    res[1] = res[1] / dataset.size
    res[2] = res[2] / dataset.size
    print(res)

    return res
end

function wikiqaSimAttenCnn:save(path, config, result, epoch)
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
    for _, vals in pairs(result) do
        for _, val in pairs(vals) do
            file:write(val .. ', ')
        end
    end
    file:write('\n')

    file:close()

    if result[1][1] > self.best_score then
        self.best_score  = result[1][1]
        self.best_params:copy(self.params)
        torch.save(paraBestPath, {params = self.params,config = config})
    end
    torch.save(paraPath, {params = self.params, config = config})
end

function wikiqaSimAttenCnn:load(path)
    local state = torch.load(path)
    self:__init(state.config)
    self.params:copy(state.params)
end
