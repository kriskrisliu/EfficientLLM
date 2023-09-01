import os
import torch
from tqdm import tqdm
import bitsandbytes as bnb
from copy import deepcopy
import fire
import sys
from fairscale.nn.model_parallel import initialize as fs_init
# sys.path.append("./model")
from model.meta import MetaModel
# sys.path.append("./util")
import util.misc as misc
from util.tensor_parallel import load_tensor_parallel_model_state_dict

"""
torchrun --nproc-per-node=1 --master-port 29400 convert_to_shared_weights.py \
--source=../checkpoints/llama2/Llama-2-13b/ \
--dst_dir=../checkpoints/effiLLaMA2/13b/base_weight_qkvoE4R512_ffE4R512/ \
--range_anchor=4 \
--rank=512 \
--filename=consolidated.00-of-01.model.pth \
--DEBUG=False \
--llama_config=../checkpoints/llama2/Llama-2-13b/params.json \
--format=meta_ori
"""
def main(
        source,
        dst_dir,
        range_anchor,
        rank,
        filename = "consolidated.00-of-01.model.pth",
        DEBUG = False,
        llama_config="",
        format="meta_ori"
        ):

    assert f"E{range_anchor}R{rank}" in dst_dir
    if not os.path.exists(dst_dir) and not DEBUG:
        os.makedirs(dst_dir)

    dst = os.path.join(dst_dir, filename)

    if os.path.isfile(source):
        ckpt = torch.load(source,map_location='cpu')
    elif os.path.isdir:
        class Args:
            pass
        args = Args()
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_on_itp = False
        args.dist_url = 'env://'

        misc.init_distributed_mode(args)
        fs_init.initialize_model_parallel(1)
        model = MetaModel("llama", [os.path.abspath(llama_config)], 
                          os.path.abspath("../checkpoints/llama2/Llama-2-7b/tokenizer.model"), with_visual=False)
        for name,module in model.named_modules():
            if name.endswith("wq"):
                print("wq:",module.weight.shape)
                break
        for name,module in model.named_modules():
            if name.endswith("w1"):
                print("w1:",module.weight.shape)
                break
        ckpt = load_tensor_parallel_model_state_dict(model, source, format, False)
        # import ipdb;ipdb.set_trace()
    ckpt_new = ckpt


    for ending_name in ["wq.weight","wk.weight","wv.weight","wo.weight",   
                        "feed_forward.w1.weight","feed_forward.w2.weight","feed_forward.w3.weight"]:
        wo_list = []
        w_name_list = []
        for key,val in ckpt.items():
            if key.endswith(ending_name):
                wo_list += [val.to("cuda", torch.float32)]
                w_name_list += [key]

        
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
                pbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.2e}, w:{w_base[0,0].item():.2e} grad:{w_base.grad[0,0].item():.2e}')
                optimizer.step()
            for layer_idx, name in enumerate(names):
                ckpt_new[name] = w_base.detach().to("cpu",ckpt_new[name].dtype)
                ckpt_new[name.replace("weight","lora_a.weight")] = w_lora_list[layer_idx][1].detach().to("cpu")
                ckpt_new[name.replace("weight","lora_b.weight")] = w_lora_list[layer_idx][0].detach().to("cpu")
            print(f"layer_id={group_idx}, loss={loss.item():.2e},", "names=",names)
            # print("-"*40)
    if not DEBUG:
        torch.save(ckpt_new, dst)
        import json
        with open(os.path.join(dst_dir, f"E{range_anchor}R{rank}.json"),'w') as fp:
            json.dump({
                    "lora_rank": rank, 
                    "lora_rank_feedforward":rank, 
                    "range_anchor":range_anchor, 
                    "bias_tuning": True}, 
                fp)
    print("Save to",dst)

fire.Fire(main)