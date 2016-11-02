
function share_params(cell, src)
    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                                    'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end


function MAP(ground_label, predict_label)
    local map = 0
    local map_idx = 0
    local extracted = {}

    for i = 1, ground_label:size(1) do
        if ground_label[i] ~= 0 then extracted[i] = 1 end
    end

    local val, key = torch.sort(predict_label, 1,true)
    for i = 1, key:size(1) do
        if extracted[key[i]] ~= nil then
            map_idx = map_idx + 1
            map = map + map_idx / i
        end
    end
    assert(map_idx ~= 0)
    map = map / map_idx
    return map
end

function MRR(ground_label, predict_label)
    local mrr = 0
    local map_idx = 0
    local extracted = {}

    for i = 1, ground_label:size(1) do
        if ground_label[i] ~= 0 then extracted[i] = 1 end
    end

    local val, key = torch.sort(predict_label, 1,true)
    for i = 1, key:size(1) do
        if extracted[key[i]] ~= nil then
            mrr = 1.0 / i
            break
        end
    end
    assert(mrr ~= 0)
    return mrr

end
