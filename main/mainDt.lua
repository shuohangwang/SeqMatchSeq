--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
include './init.lua'


local recordTrain, recordTest, recordDev

function worker()
    torch.setnumthreads(1)
    include './init.lua'

    local ivocab = tr:loadiVocab(opt.task)
    opt.numWords = #ivocab

    local model_class = seqmatchseq[opt.model]
    local model = model_class(opt)

    parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)
    local id = parallel.id
    local index = 0
    local fileid = 0
    local data = tr:loadData('train', opt.task)
    data.size = #data
    torch.manualSeed(opt.seed)
    local indices = torch.randperm(data.size)

    local readNewFile = true


    while true do
        m = parallel.yield()
        if m == 'break' then break end

        local recData = parallel.parent:receive()
        model.params:copy(recData)
        if m == 'train' then
            local batchData = {}
            for i = 1, opt.batch_size do
                local data_id = id+index+(i-1)*opt.num_processes
                if data_id <= data.size then
                    local batch_id = indices[data_id]
                    batchData[i] = data[batch_id]
                else
                    break
                end
            end
            index = index + opt.batch_size * opt.num_processes
            if index >= data.size then index = 0 --[[indices = torch.randperm(data.size)]] end


            if #batchData ~= 0 then
  			  model:trainDt(batchData)
                parallel.parent:send(model.grad_params)
            else
                parallel.parent:send(nil)
            end

        else
            local test_data = tr:loadData('dev', opt.task)
            local test_batch = torch.ceil(#test_data / opt.num_processes)
            local test_batch_data = {}
            local test_index = (id-1)*test_batch
            for i = 1, test_batch do
                if test_index+i <= #test_data then
                    test_batch_data[i] = test_data[test_index+i]
                end
            end
            local predictions = model:predict_datasetDt(test_batch_data)
            parallel.parent:send(predictions)
        end
        collectgarbage()
    end

end

function parent()
    parallel.print('Im the parent, my ID is: ' .. parallel.id)
    tr:init(opt)
    local model_class = seqmatchseq[opt.model]
    local model = model_class(opt)

    parallel.nfork(opt.num_processes)

    parallel.children:exec(worker)
    model.optim_state = { learningRate = opt.learning_rate }
    local train_dataset = tr:loadData('train', opt.task, opt.structure)
    train_dataset = #train_dataset
    local k_max = torch.ceil( train_dataset / (opt.batch_size*opt.num_processes) )
    for i = 1, opt.max_epochs do
        xlua.progress(1, train_dataset)
        for k = 1, k_max do
            model.grad_params:zero()
            local feval = function(x)
                for p = 1, opt.num_processes do
                    parallel.children[p]:join('train')
                    parallel.children[p]:send(model.params)
                end

                local replies = parallel.children:receive()
                for _, reply in pairs(replies) do
                    model.grad_params:add(reply)
                end
                model.grad_params:div(#replies)
                return 1, model.grad_params
            end
            optim[opt.grad](feval, model.params, model.optim_state)
            collectgarbage()
            if k ~= k_max then
                xlua.progress((k)*(opt.batch_size*opt.num_processes), train_dataset)
            else
                xlua.progress(train_dataset, train_dataset)
            end

        end
        for p = 1, opt.num_processes do
            parallel.children[p]:join('dev')
            parallel.children[p]:send(model.params)
        end
        local replies = parallel.children:receive()

        local fileP = io.open('../trainedmodel/evaluation/squad/dev_output.txt', "w")
        for _, reply in pairs(replies) do
            for _, rep in pairs(reply) do
                fileP:write(rep)
            end
        end
        fileP:close()
        sys.execute('python ../trainedmodel/evaluation/squad/txt2js.py ../data/squad/dev-v1.1.json ../trainedmodel/evaluation/squad/dev_output.txt ../trainedmodel/evaluation/squad/prediction.json')
        local recordDev = sys.execute('python ../trainedmodel/evaluation/squad/evaluate-v1.1.py ../data/squad/dev-v1.1.json ../trainedmodel/evaluation/squad/prediction.json')

        model:save('../trainedmodel/', opt, {recordDev, recordTrain}, i)

    end

    parallel.children:join('break')
    parallel.print('all processes terminated')
end
ok,err = pcall(parent)
if not ok then print(err) parallel.close() end
