UInst3D(
  (backbone): FPN3D(
    (fpn_lateral1): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral2): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral3): Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output3): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral4): Conv3d(1024, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output4): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (top_block): LastLevelP6P7(
      (p6): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (p7): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    )
    (bottom_up): GeneralRes3D(
      (res0): Cat()
      (res1): Sequential(
        (0): Sequential(
          (conv): StdConv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        )
      )
      (res2): Sequential(
        (0): Sequential(
          (pad): ConstantPad3d(padding=(1, 1, 1, 1, 1, 1), value=0)
          (pool): MaxPool3d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
          (unit01): PreActBottleneck(
            (gn1): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv1): StdConv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv2): StdConv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv3): StdConv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
            (downsample): StdConv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          )
          (unit02): PreActBottleneck(
            (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv1): StdConv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv2): StdConv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv3): StdConv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit03): PreActBottleneck(
            (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv1): StdConv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv2): StdConv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 64, eps=1e-05, affine=True)
            (conv3): StdConv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
        )
      )
      (res3): Sequential(
        (0): Sequential(
          (unit01): PreActBottleneck(
            (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv1): StdConv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv2): StdConv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv3): StdConv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
            (downsample): StdConv3d(256, 512, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
          )
          (unit02): PreActBottleneck(
            (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
            (conv1): StdConv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv2): StdConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv3): StdConv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit03): PreActBottleneck(
            (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
            (conv1): StdConv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv2): StdConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv3): StdConv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit04): PreActBottleneck(
            (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
            (conv1): StdConv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv2): StdConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 128, eps=1e-05, affine=True)
            (conv3): StdConv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
        )
      )
      (res4): Sequential(
        (0): Sequential(
          (unit01): PreActBottleneck(
            (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
            (conv1): StdConv3d(512, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv2): StdConv3d(256, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv3): StdConv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
            (downsample): StdConv3d(512, 1024, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
          )
          (unit02): PreActBottleneck(
            (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
            (conv1): StdConv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv2): StdConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv3): StdConv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit03): PreActBottleneck(
            (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
            (conv1): StdConv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv2): StdConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv3): StdConv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit04): PreActBottleneck(
            (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
            (conv1): StdConv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv2): StdConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv3): StdConv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit05): PreActBottleneck(
            (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
            (conv1): StdConv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv2): StdConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv3): StdConv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
          (unit06): PreActBottleneck(
            (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
            (conv1): StdConv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv2): StdConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
            (conv3): StdConv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (relu): ReLU(inplace=True)
          )
        )
      )
    )
  )
  (proposal_generator): FCOS3D(
    (fcos_head): FCOSHead(
      (cls_tower): Sequential(
        (0): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): GroupNorm(32, 128, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (4): GroupNorm(32, 128, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (7): GroupNorm(32, 128, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (10): GroupNorm(32, 128, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (bbox_tower): Sequential(
        (0): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): GroupNorm(32, 128, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (4): GroupNorm(32, 128, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (7): GroupNorm(32, 128, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (10): GroupNorm(32, 128, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (share_tower): Sequential()
      (cls_logits): Conv3d(128, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bbox_pred): Conv3d(128, 6, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (ctrness): Conv3d(128, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (cplness): Conv3d(128, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (scales): ModuleList(
        (0): Scale()
        (1): Scale()
        (2): Scale()
        (3): Scale()
        (4): Scale()
      )
    )
    (fcos_outputs): FCOSOutputs3D(
      (loc_loss_func): IOULoss()
    )
  )
  (mask_head): DynamicMaskHead3D()
  (mask_branch): MaskBranch3D(
    (refine): ModuleList(
      (0): Sequential(
        (0): Conv3d(128, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv3d(128, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv3d(128, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv3d(128, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (tower): Sequential(
      (0): Sequential(
        (0): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (ups): Sequential(
      (0): Conv3d(32, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  )
  (controller): Conv3d(128, 177, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
)
model buff size: 0.001MB, model param size:93.924MB
{'UInst3D': {'backbone': {'FPN3D': {'bottom_up': {'GeneralRes3D': {'res0': {'Cat': {'size': 0}},
                                                                   'res1': {'Sequential': {'0': {'Sequential': {'conv': {'StdConv3d': {'size': 263424}},
                                                                                                                'size': 263424}},
                                                                                           'size': 263424}},
                                                                   'res2': {'Sequential': {'0': {'Sequential': {'pad': {'ConstantPad3d': {'size': 0}},
                                                                                                                'pool': {'MaxPool3d': {'size': 0}},
                                                                                                                'size': 1744384,
                                                                                                                'unit01': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 16384}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 442368}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 65536}},
                                                                                                                                                'downsample': {'StdConv3d': {'size': 65536}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 512}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 512}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 512}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 591360}},
                                                                                                                'unit02': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 65536}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 442368}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 65536}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 512}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 512}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 576512}},
                                                                                                                'unit03': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 65536}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 442368}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 65536}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 512}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 512}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 576512}}}},
                                                                                           'size': 1744384}},
                                                                   'res3': {'Sequential': {'0': {'Sequential': {'size': 9590784,
                                                                                                                'unit01': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 131072}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 1769472}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'downsample': {'StdConv3d': {'size': 524288}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 2691072}},
                                                                                                                'unit02': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 1769472}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 4096}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 2299904}},
                                                                                                                'unit03': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 1769472}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 4096}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 2299904}},
                                                                                                                'unit04': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 1769472}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 262144}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 4096}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 1024}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 2299904}}}},
                                                                                           'size': 9590784}},
                                                                   'res4': {'Sequential': {'0': {'Sequential': {'size': 56692736,
                                                                                                                'unit01': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 524288}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 7077888}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'downsample': {'StdConv3d': {'size': 2097152}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 4096}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 10756096}},
                                                                                                                'unit02': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 7077888}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 8192}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 9187328}},
                                                                                                                'unit03': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 7077888}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 8192}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 9187328}},
                                                                                                                'unit04': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 7077888}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 8192}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 9187328}},
                                                                                                                'unit05': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 7077888}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 8192}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 9187328}},
                                                                                                                'unit06': {'PreActBottleneck': {'conv1': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'conv2': {'StdConv3d': {'size': 7077888}},
                                                                                                                                                'conv3': {'StdConv3d': {'size': 1048576}},
                                                                                                                                                'gn1': {'GroupNorm': {'size': 8192}},
                                                                                                                                                'gn2': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'gn3': {'GroupNorm': {'size': 2048}},
                                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                                'size': 9187328}}}},
                                                                                           'size': 56692736}},
                                                                   'size': 68291328}},
                                    'fpn_lateral1': {'Conv3d': {'size': 33280}},
                                    'fpn_lateral2': {'Conv3d': {'size': 131584}},
                                    'fpn_lateral3': {'Conv3d': {'size': 262656}},
                                    'fpn_lateral4': {'Conv3d': {'size': 524800}},
                                    'fpn_output1': {'Conv3d': {'size': 1769984}},
                                    'fpn_output2': {'Conv3d': {'size': 1769984}},
                                    'fpn_output3': {'Conv3d': {'size': 1769984}},
                                    'fpn_output4': {'Conv3d': {'size': 1769984}},
                                    'size': 79863552,
                                    'top_block': {'LastLevelP6P7': {'p6': {'Conv3d': {'size': 1769984}},
                                                                    'p7': {'Conv3d': {'size': 1769984}},
                                                                    'size': 3539968}}}},
             'controller': {'Conv3d': {'size': 2447556}},
             'mask_branch': {'MaskBranch3D': {'refine': {'ModuleList': {'0': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 256}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442624}},
                                                                        '1': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 256}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442624}},
                                                                        '2': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 256}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442624}},
                                                                        '3': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 256}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442624}},
                                                                        'size': 1770496}},
                                              'size': 1882400,
                                              'tower': {'Sequential': {'0': {'Sequential': {'0': {'Conv3d': {'size': 110592}},
                                                                                            '1': {'BatchNorm3d': {'size': 256}},
                                                                                            '2': {'ReLU': {'size': 0}},
                                                                                            'size': 110848}},
                                                                       'size': 110848}},
                                              'ups': {'Sequential': {'0': {'Conv3d': {'size': 1056}},
                                                                     'size': 1056}}}},
             'mask_head': {'DynamicMaskHead3D': {'size': 0}},
             'proposal_generator': {'FCOS3D': {'fcos_head': {'FCOSHead': {'bbox_pred': {'Conv3d': {'size': 82968}},
                                                                          'bbox_tower': {'Sequential': {'0': {'Conv3d': {'size': 1769984}},
                                                                                                        '1': {'GroupNorm': {'size': 1024}},
                                                                                                        '10': {'GroupNorm': {'size': 1024}},
                                                                                                        '11': {'ReLU': {'size': 0}},
                                                                                                        '2': {'ReLU': {'size': 0}},
                                                                                                        '3': {'Conv3d': {'size': 1769984}},
                                                                                                        '4': {'GroupNorm': {'size': 1024}},
                                                                                                        '5': {'ReLU': {'size': 0}},
                                                                                                        '6': {'Conv3d': {'size': 1769984}},
                                                                                                        '7': {'GroupNorm': {'size': 1024}},
                                                                                                        '8': {'ReLU': {'size': 0}},
                                                                                                        '9': {'Conv3d': {'size': 1769984}},
                                                                                                        'size': 7084032}},
                                                                          'cls_logits': {'Conv3d': {'size': 13828}},
                                                                          'cls_tower': {'Sequential': {'0': {'Conv3d': {'size': 1769984}},
                                                                                                       '1': {'GroupNorm': {'size': 1024}},
                                                                                                       '10': {'GroupNorm': {'size': 1024}},
                                                                                                       '11': {'ReLU': {'size': 0}},
                                                                                                       '2': {'ReLU': {'size': 0}},
                                                                                                       '3': {'Conv3d': {'size': 1769984}},
                                                                                                       '4': {'GroupNorm': {'size': 1024}},
                                                                                                       '5': {'ReLU': {'size': 0}},
                                                                                                       '6': {'Conv3d': {'size': 1769984}},
                                                                                                       '7': {'GroupNorm': {'size': 1024}},
                                                                                                       '8': {'ReLU': {'size': 0}},
                                                                                                       '9': {'Conv3d': {'size': 1769984}},
                                                                                                       'size': 7084032}},
                                                                          'cplness': {'Conv3d': {'size': 13828}},
                                                                          'ctrness': {'Conv3d': {'size': 13828}},
                                                                          'scales': {'ModuleList': {'0': {'Scale': {'size': 4}},
                                                                                                    '1': {'Scale': {'size': 4}},
                                                                                                    '2': {'Scale': {'size': 4}},
                                                                                                    '3': {'Scale': {'size': 4}},
                                                                                                    '4': {'Scale': {'size': 4}},
                                                                                                    'size': 20}},
                                                                          'share_tower': {'Sequential': {'size': 0}},
                                                                          'size': 14292536}},
                                               'fcos_outputs': {'FCOSOutputs3D': {'loc_loss_func': {'IOULoss': {'size': 0}},
                                                                                  'size': 0}},
                                               'size': 14292536}},
             'size': 98486044}}
