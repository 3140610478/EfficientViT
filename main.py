import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm, trange

base_folder = os.path.abspath(os.path.dirname(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Networks import criterion
    from Networks.SegEViT import SegEViT
    from Data.Gaofen import train_loader, val_loader, len_train, len_val
    from Log.Logger import getLogger
    from efficientvit.seg_model_zoo import create_seg_model

save_folder = os.path.abspath(os.path.join(base_folder, './save'))
model = SegEViT().to(config.device)
model.requires_grad_(True)
for f in (criterion.bce, criterion.acc, criterion.miou):
    f = f.to(config.device)
logger = getLogger("ISPRS Water-body Segmentation")


def train_epochs(model, start, end, lr=0.00001, transfer: bool=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    epoch_message = "\n[Epoch {:0>4d}]\ntrain_loss: {:.3f},\ttrain_acc: {:.3f}\ttrain_miou: {:.3f}\nval_loss  : {:.3f}\tval_acc  : {:.3f}\tval_miou  : {:.3f}\n"
    for epoch in trange(start, end, desc="Epoch         "):
        lossT, accT, miouT, lossV, accV, miouV = (0 for _ in range(6))
        # --- train the model ---
        torch.cuda.empty_cache()
        if transfer:
            model.fit()
        else:
            model.train()
        for x, y, z in tqdm(train_loader, desc="Training Batch"):
            optimizer.zero_grad()
            
            z = torch.logical_or(z, y).to(torch.float32)
            y_pre = model(x)
            y_pre = y_pre * z

            loss = criterion.bce(y_pre, y)
            acc = criterion.acc(y_pre, y)
            miou = criterion.miou(y_pre, y)

            loss.backward()
            optimizer.step()
            lossT += loss * len(y)
            accT += acc * len(y)
            miouT += miou * len(y)
        lossT = float((lossT / len_train).cpu())
        accT = float((accT / len_train).cpu())
        miouT = float((miouT / len_train).cpu())
        # --- validate the model ---
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for x, y, z in tqdm(val_loader, desc="Validating Batch"):
                
                z = torch.logical_or(z, y).to(torch.float32)
                y_pre = model(x)
                y_pre = y_pre * z
                
                loss = criterion.bce(y_pre, y)
                acc = criterion.acc(y_pre, y)
                miou = criterion.miou(y_pre, y)
                
                lossV += loss * len(y)
                accV += acc * len(y)
                miouV += miou * len(y)
        lossV = float((lossV / len_val).cpu())
        accV = float((accV / len_val).cpu())
        miouV = float((miouV / len_val).cpu())
        logger.info(epoch_message.format(
            epoch, lossT, accT, miouT, lossV, accV, miouV
        ))
        torch.save(model.state_dict(), os.path.abspath(
            os.path.join(save_folder, f'./dl_v3_epoch{epoch}.pth')))
        torch.save(model, os.path.abspath(
            os.path.join(save_folder, f'./dl_v3_epoch{epoch}.model')))


# training
if __name__ == "__main__":
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    train_epochs(
        model,
        0,
        5,
        0.0002,
        transfer=True,
    )
    train_epochs(
        model,
        5,
        20,
        0.00005,
        transfer=False,
    )
    train_epochs(
        model,
        20,
        40,
        0.00002,
        transfer=False,
    )
    train_epochs(
        model,
        40,
        60,
        0.000005,
        transfer=False,
    )



# EfficientViTSeg(
#   (backbone): EfficientViTBackbone(
#     (input_stem): OpSequential(
#       (op_list): ModuleList(
#         (0): ConvLayer(
#           (conv): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (norm): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (act): Hardswish()
#         )
#         (1): ResidualBlock(
#           (main): DSConv(
#             (depth_conv): ConvLayer(
#               (conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
#               (norm): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               (act): Hardswish()
#             )
#             (point_conv): ConvLayer(
#               (conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#               (norm): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             )
#           )
#           (shortcut): IdentityLayer()
#         )
#       )
#     )
#     (stages): ModuleList(
#       (0): OpSequential(
#         (op_list): ModuleList(
#           (0): ResidualBlock(
#             (main): MBConv(
#               (inverted_conv): ConvLayer(
#                 (conv): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (depth_conv): ConvLayer(
#                 (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
#                 (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (point_conv): ConvLayer(
#                 (conv): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               )
#             )
#           )
#           (1-2): 2 x ResidualBlock(
#             (main): MBConv(
#               (inverted_conv): ConvLayer(
#                 (conv): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (depth_conv): ConvLayer(
#                 (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
#                 (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (point_conv): ConvLayer(
#                 (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               )
#             )
#             (shortcut): IdentityLayer()
#           )
#         )
#       )
#       (1): OpSequential(
#         (op_list): ModuleList(
#           (0): ResidualBlock(
#             (main): MBConv(
#               (inverted_conv): ConvLayer(
#                 (conv): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (depth_conv): ConvLayer(
#                 (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
#                 (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (point_conv): ConvLayer(
#                 (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               )
#             )
#           )
#           (1-3): 3 x ResidualBlock(
#             (main): MBConv(
#               (inverted_conv): ConvLayer(
#                 (conv): Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (depth_conv): ConvLayer(
#                 (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
#                 (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 (act): Hardswish()
#               )
#               (point_conv): ConvLayer(
#                 (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               )
#             )
#             (shortcut): IdentityLayer()
#           )
#         )
#       )
#       (2): OpSequential(
#         (op_list): ModuleList(
#           (0): ResidualBlock(
#             (main): MBConv(
#               (inverted_conv): ConvLayer(
#                 (conv): Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1))
#                 (act): Hardswish()
#               )
#               (depth_conv): ConvLayer(
#                 (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=384)
#                 (act): Hardswish()
#               )
#               (point_conv): ConvLayer(
#                 (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               )
#             )
#           )
#           (1-4): 4 x EfficientViTBlock(
#             (context_module): ResidualBlock(
#               (main): LiteMLA(
#                 (qkv): ConvLayer(
#                   (conv): Conv2d(192, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 )
#                 (aggreg): ModuleList(
#                   (0): Sequential(
#                     (0): Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
#                     (1): Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), groups=18, bias=False)
#                   )
#                 )
#                 (kernel_func): ReLU()
#                 (proj): ConvLayer(
#                   (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                   (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 )
#               )
#               (shortcut): IdentityLayer()
#             )
#             (local_module): ResidualBlock(
#               (main): MBConv(
#                 (inverted_conv): ConvLayer(
#                   (conv): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
#                   (act): Hardswish()
#                 )
#                 (depth_conv): ConvLayer(
#                   (conv): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
#                   (act): Hardswish()
#                 )
#                 (point_conv): ConvLayer(
#                   (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                   (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 )
#               )
#               (shortcut): IdentityLayer()
#             )
#           )
#         )
#       )
#       (3): OpSequential(
#         (op_list): ModuleList(
#           (0): ResidualBlock(
#             (main): MBConv(
#               (inverted_conv): ConvLayer(
#                 (conv): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
#                 (act): Hardswish()
#               )
#               (depth_conv): ConvLayer(
#                 (conv): Conv2d(768, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=768)
#                 (act): Hardswish()
#               )
#               (point_conv): ConvLayer(
#                 (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               )
#             )
#           )
#           (1-6): 6 x EfficientViTBlock(
#             (context_module): ResidualBlock(
#               (main): LiteMLA(
#                 (qkv): ConvLayer(
#                   (conv): Conv2d(384, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 )
#                 (aggreg): ModuleList(
#                   (0): Sequential(
#                     (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#                     (1): Conv2d(1152, 1152, kernel_size=(1, 1), stride=(1, 1), groups=36, bias=False)
#                   )
#                 )
#                 (kernel_func): ReLU()
#                 (proj): ConvLayer(
#                   (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                   (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 )
#               )
#               (shortcut): IdentityLayer()
#             )
#             (local_module): ResidualBlock(
#               (main): MBConv(
#                 (inverted_conv): ConvLayer(
#                   (conv): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
#                   (act): Hardswish()
#                 )
#                 (depth_conv): ConvLayer(
#                   (conv): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
#                   (act): Hardswish()
#                 )
#                 (point_conv): ConvLayer(
#                   (conv): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                   (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                 )
#               )
#               (shortcut): IdentityLayer()
#             )
#           )
#         )
#       )
#     )
#   )
#   (head): SegHead(
#     (input_ops): ModuleList(
#       (0): OpSequential(
#         (op_list): ModuleList(
#           (0): ConvLayer(
#             (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#           (1): UpSampleLayer()
#         )
#       )
#       (1): OpSequential(
#         (op_list): ModuleList(
#           (0): ConvLayer(
#             (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#           (1): UpSampleLayer()
#         )
#       )
#       (2): ConvLayer(
#         (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (middle): OpSequential(
#       (op_list): ModuleList(
#         (0-2): 3 x ResidualBlock(
#           (main): MBConv(
#             (inverted_conv): ConvLayer(
#               (conv): Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#               (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               (act): Hardswish()
#             )
#             (depth_conv): ConvLayer(
#               (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
#               (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#               (act): Hardswish()
#             )
#             (point_conv): ConvLayer(
#               (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#               (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             )
#           )
#           (shortcut): IdentityLayer()
#         )
#       )
#     )
#     (output_ops): ModuleList(
#       (0): OpSequential(
#         (op_list): ModuleList(
#           (0): ConvLayer(
#             (conv): Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (act): Hardswish()
#           )
#           (1): ConvLayer(
#             (conv): Conv2d(384, 19, kernel_size=(1, 1), stride=(1, 1))
#           )
#         )
#       )
#     )
#   )
# )