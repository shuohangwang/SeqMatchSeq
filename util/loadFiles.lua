--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
local stringx = require 'pl.stringx'
require 'debug'
require 'paths'

tr = {}
tr.__index = tr
function tr:init(opt)
    if not paths.filep("../data/".. opt.task .."/vocab.t7") then
        self:buildVocab(opt.task)
    end

    if not paths.filep("../data/".. opt.task .."/sequence/train.t7") then
        if opt.task == 'snil' then
            tr:buildData('dev', opt.task)
            tr:buildData('test', opt.task)
            tr:buildData('train', opt.task)
        else
            tr:buildData('all', opt.task)
        end
    end

    if not paths.filep("../data/".. opt.task .."/initEmb.t7") then
        self:buildVacab2Emb(opt)
    end

end

function tr:loadVocab(task)
    return  torch.load("../data/".. task .."/vocab.t7")
end

function tr:loadUnUpdateVocab(task)
    return  torch.load("../data/".. task .."/unUpdateVocab.t7")
end

function tr:loadiVocab(task)
    return torch.load("../data/"..task.."/ivocab.t7")
end

function tr:loadVacab2Emb(task)
    print("Loading embedding ...")
    return torch.load("../data/"..task.."/initEmb.t7")
end

function tr:loadData(filename, task)
    print("Loading data "..filename.."...")
    return torch.load("../data/"..task.."/sequence/"..filename..".t7")
end

function tr:buildVocab(task)
    print ("Building vocab dict ...")
    if task == 'snli' then
        local tokens = {}
        local index = {}
        local count = 1
        local filenames = {"../data/"..task.."/sequence/train.txt", "../data/"..task.."/sequence/dev.txt", "../data/"..task.."/sequence/test.txt"}

        for _, filename in pairs(filenames) do
            for line in io.lines(filename) do
                local sents = stringx.split(line:lower(), '\t')
                for s = 1, 2 do
                    local words = stringx.split(sents[s], ' ')
                    for _, word in pairs(words) do
                        if tokens[word] == nil then
                            tokens[word] = count
                            index[count] = word

                            count = count + 1
                        end
                    end
                end
            end
        end

        torch.save("../data/"..task.."/vocab.t7", tokens)
        torch.save("../data/"..task.."/ivocab.t7", index)

        local file = io.open("../data/"..task.."/vocab.txt","w")
        for word, i in pairs(tokens) do
            file:write(word .. ' ' .. i .. '\n')
        end
        file:close()
    elseif task == 'squad' then
        local filenames = {dev="../data/"..task.."/sequence/dev.txt", train="../data/"..task.."/sequence/train.txt"}
        local vocab = {}
        local ivocab = {}
        local a_vocab = {}
        local a_ivocab = {}
        for _, filename in pairs(filenames) do
            for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                for j = 1, 2 do
                    local words = stringx.split(divs[j], ' ')
                    for i = 1, #words do
                        if vocab[words[i]] == nil then
                            vocab[words[i]] = #ivocab + 1
                            ivocab[#ivocab + 1] = words[i]
                        end
                    end
                end
            end
        end
        torch.save("../data/"..task.."/vocab.t7", vocab)
        torch.save("../data/"..task.."/ivocab.t7", ivocab)
    elseif task == 'wikiqa' then
        local vocab = {}
        local ivocab = {}
        vocab['NULL'] = 1
        ivocab[1] = 'NULL'
        local filenames = {"../data/"..task.."/WikiQACorpus/WikiQA-train.txt","../data/"..task.."/WikiQACorpus/WikiQA-dev.txt","../data/"..task.."/WikiQACorpus/WikiQA-test.txt"}
        for _, filename in pairs(filenames) do
            for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                for m = 1, 2 do
                    local words = stringx.split(divs[m]:lower(), ' ')
                    for i = 1, #words do
                        if vocab[words[i]] == nil then
                            vocab[words[i]] = #ivocab + 1
                            ivocab[#ivocab + 1] = words[i]
                        end
                    end
                end
            end
        end
        print(#ivocab)
        torch.save("../data/"..task.."/vocab.t7", vocab)
        torch.save("../data/"..task.."/ivocab.t7", ivocab)
    else
        error('The specified task is not supported yet!')
    end
end


function tr:buildVacab2Emb(opt)

    local vocab = self:loadVocab(opt.task)
    local ivocab = self:loadiVocab(opt.task)
    local emb = torch.randn(#ivocab, opt.wvecDim) * 0.05
    if opt.task ~= 'snli' then emb:zero() end

    print ("Loading ".. opt.preEmb .. " ...")
    local file
    if opt.preEmb == 'glove' then
        file = io.open("../data/"..opt.preEmb.."/glove.840B.300d.txt", 'r')
    end

    local count = 0
    local embRec = {}
    while true do
        local line = file:read()

        if line == nil then break end
        vals = stringx.split(line, ' ')
        if vocab[vals[1]] ~= nil then
            for i = 2, #vals do
                emb[vocab[vals[1]]][i-1] = tonumber(vals[i])
            end
            embRec[vocab[vals[1]]] = 1
            count = count + 1
            if count == #ivocab then
                break
            end
        end
    end
    print("Number of words not appear in ".. opt.preEmb .. ": "..(#ivocab-count) )
    if opt.task == 'snli' then
        self:initialUNK(embRec, emb, opt)
    end
    torch.save("../data/"..opt.task.."/initEmb.t7", emb)
    torch.save("../data/"..opt.task.."/unUpdateVocab.t7", embRec)
end

function tr:initialUNK(embRec, emb, opt)
    print("Initializing not appeared words ...")
    local windowSize = 4
    local numRec = {}
    local filenames = {'train', 'dev', 'test'}
    local sentsnames = {'lsents', 'rsents'}
    for _, filename in pairs(filenames) do
        local data = tr:loadData(filename, opt.task)
        for _, sentsname in pairs(sentsnames) do
            for _, sent in pairs(data[sentsname]) do
                for i = 1, sent:size(1) do
                    local word = sent[i]
                    if embRec[word] == nil then
                        if numRec[word] == nil then
                            numRec[word] = 0
                        end

                        local count = 0
                        for j = -windowSize, windowSize do
                            if i + j <= sent:size(1) and i + j >= 1 and embRec[sent[i+j]] ~= nil then
                                emb[word] = emb[word] + emb[sent[i+j]]
                                count = count + 1
                            end
                        end
                        numRec[word] = numRec[word] + count
                    end
                end
            end
        end
    end
    for k, v in pairs(numRec) do
        if v ~= 0 then
            emb[k] = emb[k] / (v+1)
        end
        --print(v)
    end
end

function tr:buildData(filename, task)
    local trees = {}
    local lines = {}
    local dataset = {}
    idx = 1
    vocab = tr:loadVocab(task)
    print ("Building "..task.." "..filename.." data ...")

    if task == 'snli' then
        dataset.lsents = {}
        dataset.rsents = {}
        dataset.labels = {}

        for line in io.lines("../data/"..task.."/sequence/"..filename..".txt") do
            local sents = stringx.split(line:lower(), '\t')
            for s = 1, 2 do
                local words = stringx.split(sents[s], ' ')
                local wordsIdx = torch.IntTensor(#words)

                for i = 1, #words do
                    local wordIdx = vocab[words[i]]
                    assert(wordIdx ~= nil, 'rebuild vocab.t7')
                    wordsIdx[i] = wordIdx
                end

                if s == 1 then
                    dataset.lsents[#dataset.lsents + 1] = wordsIdx
                else
                    dataset.rsents[#dataset.rsents + 1] = wordsIdx
                end

            end
            dataset.labels[#dataset.labels+1] = tonumber(sents[3])
        end
        dataset.size = #dataset.lsents
        assert(#dataset.lsents == #dataset.labels)
        torch.save("../data/"..task.."/sequence/"..filename..".t7", dataset)
    elseif task == 'squad' then
        local filenames = {dev="../data/"..task.."/sequence/dev.txt", train="../data/"..task.."/sequence/train.txt"}
        for folder, filename in pairs(filenames) do
            local data = {}
            for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                local instance = {}
                for j = 1, 2 do
                    local words = stringx.split(divs[j], ' ')
                    instance[j] = torch.IntTensor(#words)
                    for i = 1, #words do
                        instance[j][i] = vocab[ words[i] ]
                    end
                end
                if folder == 'train' then
                    local pos = stringx.split(stringx.strip(divs[3]), ' ')
                    instance[3] = torch.IntTensor(#pos+1)
                    for i = 1, #pos do
                        instance[3][i] = tonumber(pos[i])
                    end
                    instance[3][#pos+1] = instance[1]:size(1)+1
                end
                data[#data+1] = instance
            end
            torch.save("../data/"..task.."/sequence/"..folder..'.t7', data)
        end
    elseif task == 'wikiqa' then
        local filenames = {train='../data/'..task..'/WikiQACorpus/WikiQA-train.txt',dev='../data/'..task..'/WikiQACorpus/WikiQA-dev.txt',test='../data/'..task..'/WikiQACorpus/WikiQA-test.txt'}
        for folder, filename in pairs(filenames) do
            local data = {}
            local instance = {}
            local candidates = {}
            local labels = {}
            for line in io.lines(filename) do
                local divs = stringx.split(line:lower(), '\t')
                if instance[1] ~= divs[1] then
                    if #instance ~= 0 then
                        labels = torch.FloatTensor(labels)
                        if labels:sum() ~= 0 then
                            local words = stringx.split(instance[1], ' ')
                            for i = 1, #words do words[i] = vocab[words[i]] end
                            instance[1] = torch.LongTensor(words)
                            instance[2] = candidates
                            instance[3] = labels:div(labels:sum())
                            data[#data + 1] = instance
                        end
                    end
                    instance = {divs[1]}
                    candidates = {}
                    labels = {}
                end
                local words = stringx.split(divs[2], ' ')
                local cand = {}
                for i = 1, #words do
                    cand[i] = vocab[words[i]]
                end
                candidates[#candidates+1] = torch.LongTensor(cand)
                labels[#labels + 1] = tonumber(divs[3])
            end
            torch.save("../data/"..task.."/sequence/"..folder..'.t7', data)
        end
    end
    return dataset
end


return tr
