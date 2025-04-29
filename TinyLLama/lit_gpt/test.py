import xformers
import xformers.ops
import torch
DualGemmSiluOp = xformers.ops.common.get_xformers_operator(
        "dual_gemm_silu_identity_mul"
    )
N, M, H = (128, 128, 256)
dtype=torch.float16
device="cuda"
#device="cpu"
x = torch.randn([N, M], device=device, dtype=dtype, requires_grad=False)
w1 = torch.randn([H, M], device=device, dtype=dtype, requires_grad=False)
w2 = torch.randn([H, M], device=device, dtype=dtype, requires_grad=False)

b1 = None
b2 = None
x1, x2, x4 = DualGemmSiluOp(x, w1, b1, w2, b2)

# import torch

# from xformers.components import MultiHeadDispatch
# from xformers.components.attention import BlockSparseAttention

# BATCH = 2
# HEADS = 8
# SEQ = 2048
# EMB = 1024
# BLOCK_SIZE = 32
# DROPOUT = 0.1
# dtype = torch.float16


# # Let's try out a causal mask, but really it could be anything "block sparse enough"
# causal_mask = torch.tril(torch.ones((SEQ, SEQ), device=torch.device("cuda"), dtype=dtype))
# #causal_mask=None
# blocks = SEQ // BLOCK_SIZE
# causal_layout = torch.tril(torch.ones([HEADS, blocks, blocks]))

# # Let's build our blocksparse attention. Please note that the layout can be
# # [SEQ//BLOCK_SIZE, SEQ//BLOCK_SIZE] or  [HEADS, SEQ//BLOCK_SIZE, SEQ//BLOCK_SIZE]
# # so that _you can pass a different layout per head_
# attention = BlockSparseAttention(layout=causal_layout, block_size=BLOCK_SIZE, dropout=DROPOUT, num_heads=HEADS)

# # Out of commodity, let's build our multihead attention now
# # "multi_head" will be responsible for the forward
# multi_head = (
#     MultiHeadDispatch(
#         seq_len=SEQ,
#         dim_model=EMB,
#         residual_dropout=DROPOUT,
#         num_heads=HEADS,
#         attention=attention,
#     )
#     .cuda()
#     .half()
# )

# # Now FW some random data
# # Note that passing a per-coefficient mask makes it possible to remove extra coefficients,
# # which where required by the blockification
# query = torch.randn((BATCH, SEQ, EMB), requires_grad=True, device=torch.device("cuda"), dtype=dtype)

# # Self attention in this particular example, no limitations really
# #att_val = multi_head(query=query, key=query, value=query, att_mask=causal_mask)
# print(query)
# att_val = multi_head(query=query, key=query, value=query)
# print(att_val)

# #########################################
# # Bonus: compare the memory use vs dense:
# def mem_use(fn, kwargs, title):
#     # bookeeping
#     import time

#     start = time.time()
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()

#     # actually run the function
#     fn(**kwargs)
#     torch.cuda.synchronize()
#     stop = time.time()

#     # now report
#     max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
#     print(f"{title} - Peak memory use: {max_memory}MB - {round((stop-start)*1e6)/1e3}ms")


# pytorch_multihead = torch.nn.MultiheadAttention(
#     EMB, HEADS, batch_first=True, device=torch.device("cuda"), dtype=torch.float16
# )

# mem_use(multi_head, {"query": query, "key": query, "value": query, "att_mask": causal_mask}, "Blocksparse")
# mem_use(pytorch_multihead, {"query": query, "key": query, "value": query, "attn_mask": causal_mask}, "PyTorch")





# import torch
# from torch import nn
# from lightning import Fabric
# from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
# # from torchinfo import summary



# def train(num_epochs,model,optimizer,data,target,fabric):
#     model.train()
#     data=fabric.to_device(data)
#     target=fabric.to_device(target)
#     #data=data.to(fabric.device)
#     #target=target.to(fabric.device)
#     print("fabric.device and local_rank and torch local rank:",fabric.device,fabric.local_rank,torch.distributed.get_rank())# 这三个是一个东西
#     for epoch in range(num_epochs):
#         out=model(data)
#         loss = torch.nn.MSELoss()(out,target)
#         optimizer.zero_grad()
#         fabric.backward(loss)
#         optimizer.step()
#         print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | train loss:{loss}") #会打印出每个GPU上的loss
#         all_loss=fabric.all_gather(loss) #获取所有loss,这个是一样大的，GPU个loss
#         print(all_loss)
#     #保存模型
#     state={"model":model,"optimizer":optimizer,"iter":epoch+1}
#     fabric.save("checkpoint.ckpt",state)
    
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.conv=nn.Conv2d(3,5,3,1)
#         self.bn = nn.BatchNorm2d(5)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.flat = nn.Flatten()
#         self.fc = nn.Linear(5, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.avg_pool(x)
#         x = self.flat(x)
#         x = self.fc(x)        
#         return x
# if __name__=="__main__":
    
#     strategy = FSDPStrategy(
#                 auto_wrap_policy={SimpleModel},
#                 activation_checkpointing_policy=None,
#                 state_dict_type="full",
#                 limit_all_gathers=True,
#                 cpu_offload=False,
#             )
#     #fabric = Fabric(accelerator="cuda",devices=[0,1],strategy="ddp",precision='16-mixed')
#     fabric = Fabric(accelerator="cuda",devices=[0,1,2,3,4,5,6,7],strategy=strategy,precision='16-mixed')
#     fabric.launch()
    
#     fabric.seed_everything()
    
#     #初始化模型
#     model = SimpleModel()
#     fabric.print(f"before setup model,state dict:")#只在GPU0上打印
#     #fabric.print(summary(model,input_size=(1,3,8,8)))
#     #fabric.print(model.state_dict().keys())
#     fabric.print("*****************************************************************")
#     optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#     if fabric.world_size>1:
#         model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         fabric.print(f"after convert bn to sync bn,state dict:")
#         #fabric.print(summary(model,input_size=(1,3,8,8)))
#         print(f"after convert bn to sync bn device:{fabric.device} conv.weight.device:{model.conv.weight.device}")
#         #fabric.print(model.state_dict().keys())
#         fabric.print("*****************************************************************")
#     model,optimizer=fabric.setup(model,optimizer)
#     print(f"after setup device:{fabric.device} conv.weight.device:{model.conv.weight.device}")
#     fabric.print(f"after setup model,model state dict:")
#     #fabric.print(summary(model,input_size=(1,3,8,8)))
#     #fabric.print(model.state_dict().keys())
#     #设置模拟数据(如果是dataloader那么除了torch.utils.data.DistributedSampler外的其它部分)
#     data= torch.rand(5,3,8,8)
#     target=torch.rand(5,1)
#     #开始训练
#     epoch=100
#     train(epoch,model,optimizer,data,target,fabric)

# import xformers
# import xformers.ops
# import torch
# DualGemmSiluOp = xformers.ops.common.get_xformers_operator(
#         "dual_gemm_silu_identity_mul"
#     )
# #N, M, H = (2048, 2048, 5632)
# N, M, H = (128, 128, 256)
# dtype=torch.float16
# device="cuda"
# #device="cpu"
# x = torch.randn([N, M], device=device, dtype=dtype, requires_grad=False)
# w1 = torch.randn([H, M], device=device, dtype=dtype, requires_grad=False)
# w2 = torch.randn([H, M], device=device, dtype=dtype, requires_grad=False)

# b1 = None
# b2 = None
# x1, x2, x4 = DualGemmSiluOp(x, w1, b1, w2, b2)

# from xformers.ops import SwiGLU
# import torch
# #x=SwiGLU(in_features=100,hidden_features=20).cuda()
# x=SwiGLU(100,20, bias=False, _pack_weights=False).cuda()
# data=torch.rand((20,100)).cuda()
# y=x(data)
# print(y)