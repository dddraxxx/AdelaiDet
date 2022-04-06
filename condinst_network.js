UInst3D(
  (backbone): FPN3D(
    (fpn_lateral0): Conv3d(32, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output0): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral1): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral2): Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral3): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output3): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral4): Conv3d(320, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output4): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (top_block): LastLevelP6P7(
      (p6): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (p7): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    )
    (bottom_up): UNETD(
      (conv_blocks_context): ModuleList(
        (0): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
            (1): ConvDropoutNormNonlin(
              (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
          )
        )
        (1): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
            (1): ConvDropoutNormNonlin(
              (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
          )
        )
        (2): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
            (1): ConvDropoutNormNonlin(
              (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
          )
        )
        (3): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
            (1): ConvDropoutNormNonlin(
              (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
          )
        )
        (4): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(256, 320, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
            (1): ConvDropoutNormNonlin(
              (conv): Conv3d(320, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
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
      (4): Sequential(
        (0): Conv3d(128, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (tower): Sequential()
    (ups): Sequential(
      (0): Conv3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
    (seg_head): Sequential(
      (0): Sequential(
        (0): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (logits): Conv3d(32, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  )
  (controller): Conv3d(128, 369, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
)
model buff size: 0.014MB, model param size:65.313MB
{'UInst3D': {'backbone': {'FPN3D': {'bottom_up': {'UNETD': {'conv_blocks_context': {'ModuleList': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 3584}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 256}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 3840}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 110720}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 256}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 110976}},
                                                                                                                                                         'size': 114816}},
                                                                                                                               'size': 114816}},
                                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 221440}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 221952}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 442624}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 443136}},
                                                                                                                                                         'size': 665088}},
                                                                                                                               'size': 665088}},
                                                                                                   '2': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 885248}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 886272}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 1769984}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 1771008}},
                                                                                                                                                         'size': 2657280}},
                                                                                                                               'size': 2657280}},
                                                                                                   '3': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 3539968}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 3542016}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 7078912}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 7080960}},
                                                                                                                                                         'size': 10622976}},
                                                                                                                               'size': 10622976}},
                                                                                                   '4': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 8848640}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 2560}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 8851200}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 11060480}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 2560}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 11063040}},
                                                                                                                                                         'size': 19914240}},
                                                                                                                               'size': 19914240}},
                                                                                                   'size': 33974400}},
                                                            'size': 33974400}},
                                    'fpn_lateral0': {'Conv3d': {'size': 16896}},
                                    'fpn_lateral1': {'Conv3d': {'size': 33280}},
                                    'fpn_lateral2': {'Conv3d': {'size': 66048}},
                                    'fpn_lateral3': {'Conv3d': {'size': 131584}},
                                    'fpn_lateral4': {'Conv3d': {'size': 164352}},
                                    'fpn_output0': {'Conv3d': {'size': 1769984}},
                                    'fpn_output1': {'Conv3d': {'size': 1769984}},
                                    'fpn_output2': {'Conv3d': {'size': 1769984}},
                                    'fpn_output3': {'Conv3d': {'size': 1769984}},
                                    'fpn_output4': {'Conv3d': {'size': 1769984}},
                                    'size': 46776448,
                                    'top_block': {'LastLevelP6P7': {'p6': {'Conv3d': {'size': 1769984}},
                                                                    'p7': {'Conv3d': {'size': 1769984}},
                                                                    'size': 3539968}}}},
             'controller': {'Conv3d': {'size': 5102532}},
             'mask_branch': {'MaskBranch3D': {'logits': {'Conv3d': {'size': 132}},
                                              'refine': {'ModuleList': {'0': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
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
                                                                        '4': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 256}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442624}},
                                                                        'size': 2213120}},
                                              'seg_head': {'Sequential': {'0': {'Sequential': {'0': {'Conv3d': {'size': 110592}},
                                                                                               '1': {'BatchNorm3d': {'size': 256}},
                                                                                               '2': {'ReLU': {'size': 0}},
                                                                                               'size': 110848}},
                                                                          'size': 110848}},
                                              'size': 2328324,
                                              'tower': {'Sequential': {'size': 0}},
                                              'ups': {'Sequential': {'0': {'Conv3d': {'size': 4224}},
                                                                     'size': 4224}}}},
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
                                                                          'ctrness': {'Conv3d': {'size': 13828}},
                                                                          'scales': {'ModuleList': {'0': {'Scale': {'size': 4}},
                                                                                                    '1': {'Scale': {'size': 4}},
                                                                                                    '2': {'Scale': {'size': 4}},
                                                                                                    '3': {'Scale': {'size': 4}},
                                                                                                    '4': {'Scale': {'size': 4}},
                                                                                                    'size': 20}},
                                                                          'share_tower': {'Sequential': {'size': 0}},
                                                                          'size': 14278708}},
                                               'fcos_outputs': {'FCOSOutputs3D': {'loc_loss_func': {'IOULoss': {'size': 0}},
                                                                                  'size': 0}},
                                               'size': 14278708}},
             'size': 68486012}}
