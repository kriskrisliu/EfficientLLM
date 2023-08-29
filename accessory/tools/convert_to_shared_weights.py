import os
import torch
from tqdm import tqdm
import bitsandbytes as bnb
from copy import deepcopy

source = "../checkpoints/llama2/Llama-2-7b/consolidated.00.pth"
dst_dir = "/data/liuyijiang/mmlab/EfficientLLM/checkpoints/effiLLaMA2/base_weight_qkvo512_ff512"
filename = "consolidated.00.pth"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

dst = os.path.join(dst_dir, filename)

ckpt = torch.load(source,map_location='cpu')
ckpt_new = deepcopy(ckpt)


for ending_name in ["wq.weight","wk.weight","wv.weight","wo.weight",   
                    "feed_forward.w1.weight","feed_forward.w2.weight","feed_forward.w3.weight"]:
    wo_list = []
    w_name_list = []
    for key,val in ckpt.items():
        if key.endswith(ending_name):
            wo_list += [val.to("cuda", torch.float32)]
            w_name_list += [key]

    rank = 512
    range_anchor = 2
    # print(f"(In a group) full params:{4096*4096*range_anchor}, retained params:{4096*rank*2*range_anchor+4096}, reduced:{(4096*rank*2*range_anchor+4096)/(4096*4096*range_anchor):.3f}")

    for group_idx in range(0,len(wo_list),range_anchor):
        w_base = torch.nn.Parameter(torch.empty_like(wo_list[group_idx])).cuda()
        torch.nn.init.xavier_normal_(w_base)
        w_lora_list = []
        for ii in range(range_anchor):
            lora1 = torch.nn.Parameter(torch.empty_like(wo_list[0][:,:rank]),requires_grad=True)
            lora2 = torch.nn.Parameter(torch.empty_like(wo_list[0][:rank,:]),requires_grad=True)
            torch.nn.init.xavier_normal_(lora1)
            torch.nn.init.xavier_normal_(lora2)
            w_lora_list +=[(lora1.cuda(), lora2.cuda())]

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
            ckpt_new[name] = w_base.to(ckpt_new[name].dtype)
            ckpt_new[name.replace("weight","lora_a.weight")] = w_lora_list[layer_idx][1]
            ckpt_new[name.replace("weight","lora_b.weight")] = w_lora_list[layer_idx][0]
        print(group_idx,loss, names)
        print("-"*40)
torch.save(ckpt_new, dst)
print("Save to",dst)