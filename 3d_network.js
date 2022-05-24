UInst3D(
  (backbone): FPN3D(
    (fpn_lateral1): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral2): Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral3): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output3): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral4): Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output4): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (top_block): LastLevelP6P7(
      (p6): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (p7): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    )
    (bottom_up): VResNet(
      (res0): Cat()
      (res1): Sequential(
        (0): BasicStem(
          (0): Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False)
          (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): Sequential(
          (0): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
          )
          (1): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
          )
        )
      )
      (res2): Sequential(
        (0): Sequential(
          (0): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
              (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
          )
        )
      )
      (res3): Sequential(
        (0): Sequential(
          (0): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
              (1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
          )
        )
      )
      (res4): Sequential(
        (0): Sequential(
          (0): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
              (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Sequential(
              (0): Conv3DSimple(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (conv2): Sequential(
              (0): Conv3DSimple(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              (1): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
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
model buff size: 0.038MB, model param size:154.878MB
{'UInst3D': {'backbone': {'FPN3D': {'bottom_up': {'VResNet': {'res0': {'Cat': {'size': 0}},
                                                              'res1': {'Sequential': {'0': {'BasicStem': {'0': {'Conv3d': {'size': 112896}},
                                                                                                          '1': {'BatchNorm3d': {'size': 512}},
                                                                                                          '2': {'ReLU': {'size': 0}},
                                                                                                          'size': 113408}},
                                                                                      '1': {'Sequential': {'0': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 442368}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 442880}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 442368}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                         'size': 442880}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 885760}},
                                                                                                           '1': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 442368}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 442880}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 442368}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                         'size': 442880}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 885760}},
                                                                                                           'size': 1771520}},
                                                                                      'size': 1884928}},
                                                              'res2': {'Sequential': {'0': {'Sequential': {'0': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 884736}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 885760}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 1769472}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                         'size': 1770496}},
                                                                                                                                'downsample': {'Sequential': {'0': {'Conv3d': {'size': 32768}},
                                                                                                                                                              '1': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                              'size': 33792}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 2690048}},
                                                                                                           '1': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 1769472}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 1770496}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 1769472}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                         'size': 1770496}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 3540992}},
                                                                                                           'size': 6231040}},
                                                                                      'size': 6231040}},
                                                              'res3': {'Sequential': {'0': {'Sequential': {'0': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 3538944}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 3540992}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 7077888}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                         'size': 7079936}},
                                                                                                                                'downsample': {'Sequential': {'0': {'Conv3d': {'size': 131072}},
                                                                                                                                                              '1': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                              'size': 133120}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 10754048}},
                                                                                                           '1': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 7077888}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 7079936}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 7077888}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                         'size': 7079936}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 14159872}},
                                                                                                           'size': 24913920}},
                                                                                      'size': 24913920}},
                                                              'res4': {'Sequential': {'0': {'Sequential': {'0': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 14155776}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 4096}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 14159872}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 28311552}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 4096}},
                                                                                                                                                         'size': 28315648}},
                                                                                                                                'downsample': {'Sequential': {'0': {'Conv3d': {'size': 524288}},
                                                                                                                                                              '1': {'BatchNorm3d': {'size': 4096}},
                                                                                                                                                              'size': 528384}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 43003904}},
                                                                                                           '1': {'BasicBlock': {'conv1': {'Sequential': {'0': {'Conv3DSimple': {'size': 28311552}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 4096}},
                                                                                                                                                         '2': {'ReLU': {'size': 0}},
                                                                                                                                                         'size': 28315648}},
                                                                                                                                'conv2': {'Sequential': {'0': {'Conv3DSimple': {'size': 28311552}},
                                                                                                                                                         '1': {'BatchNorm3d': {'size': 4096}},
                                                                                                                                                         'size': 28315648}},
                                                                                                                                'relu': {'ReLU': {'size': 0}},
                                                                                                                                'size': 56631296}},
                                                                                                           'size': 99635200}},
                                                                                      'size': 99635200}},
                                                              'size': 132665088}},
                                    'fpn_lateral1': {'Conv3d': {'size': 33280}},
                                    'fpn_lateral2': {'Conv3d': {'size': 66048}},
                                    'fpn_lateral3': {'Conv3d': {'size': 131584}},
                                    'fpn_lateral4': {'Conv3d': {'size': 262656}},
                                    'fpn_output1': {'Conv3d': {'size': 1769984}},
                                    'fpn_output2': {'Conv3d': {'size': 1769984}},
                                    'fpn_output3': {'Conv3d': {'size': 1769984}},
                                    'fpn_output4': {'Conv3d': {'size': 1769984}},
                                    'size': 143778560,
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
             'size': 162401052}}
