import torch
from tqdm import tqdm
from util import misc

def reconstruct(local_state_dict, model_args):
    
    lora_rank = model_args.lora_rank
    lora_rank_feedforward = model_args.lora_rank_feedforward
    range_anchor = model_args.range_anchor

    if misc.is_main_process():

        for ending_name in ["wq.weight","wk.weight","wv.weight","wo.weight",   
                            "feed_forward.w1.weight","feed_forward.w2.weight","feed_forward.w3.weight"]:
            wo_list = []
            w_name_list = []
            for key,val in local_state_dict.items():
                if key.endswith(ending_name):
                    wo_list += [val.to("cuda:0", torch.float32)]
                    w_name_list += [key]
            rank = lora_rank if ending_name in ["wq.weight","wk.weight","wv.weight","wo.weight"] else lora_rank_feedforward
            
            for group_idx in range(0,len(wo_list),range_anchor):
                # if (group_idx+range_anchor) > (len(wo_list)-1):
                #     range_anchor = len(wo_list) - group_idx - 1
                w_base = torch.nn.Parameter(torch.empty_like(wo_list[group_idx])).to("cuda:0")
                torch.nn.init.xavier_normal_(w_base)
                w_lora_list = []
                for ii in range(range_anchor):
                    lora1 = torch.nn.Parameter(torch.empty_like(wo_list[0][:,:rank]),requires_grad=True)
                    lora2 = torch.nn.Parameter(torch.empty_like(wo_list[0][:rank,:]),requires_grad=True)
                    torch.nn.init.xavier_normal_(lora1)
                    torch.nn.init.xavier_normal_(lora2)
                    w_lora_list +=[(lora1.to("cuda:0"), lora2.to("cuda:0"))]

                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.SGD([w_lora_list[ii][0] for ii in range(range_anchor)]+[w_lora_list[ii][1] for ii in range(range_anchor)]+[w_base], lr=500000)

                pbar = tqdm([ll for ll in range(1000)], desc=f'Training group_idx={group_idx}', leave=True)
                for epoch in pbar:
                    optimizer.zero_grad()           # Zero the gradients
                    loss = 0
                    names = w_name_list[group_idx:group_idx+range_anchor]
                    for idx, ww in enumerate(wo_list[group_idx:group_idx+range_anchor]):
                        w_approx = w_base + w_lora_list[idx][0] @ w_lora_list[idx][1]
                        loss += criterion(w_approx, ww)  # Calculate loss
                    loss.backward()
                    pbar.set_description(f'Training - Epoch {epoch+1}, Loss: {loss.item():.2e}, w:{w_base[0,0].item():.2e} grad:{w_base.grad[0,0].item():.2e}')
                    optimizer.step()
                for layer_idx, name in enumerate(names):
                    local_state_dict[name] = w_base.to(local_state_dict[name].dtype)
                    local_state_dict[name.replace("weight","lora_a.weight")] = w_lora_list[layer_idx][1]
                    local_state_dict[name.replace("weight","lora_b.weight")] = w_lora_list[layer_idx][0]
                print(group_idx,loss, names)
                print("-"*40)
            # import ipdb;ipdb.set_trace()