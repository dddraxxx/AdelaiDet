UInst3D(
  (backbone): FPN3D(
    (fpn_lateral1): Conv3d(32, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral2): Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral3): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (fpn_lateral4): Conv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (fpn_output4): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (top_block): LastLevelP6P7(
      (p6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (p7): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    )
    (bottom_up): UNETD(
      (conv_blocks_context): ModuleList(
        (0): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
            (1): ConvDropoutNormNonlin(
              (conv): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (instnorm): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            )
          )
        )
        (1): StackedConvLayers(
          (blocks): Sequential(
            (0): ConvDropoutNormNonlin(
              (conv): Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
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
        (2): StackedConvLayers(
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
        (3): StackedConvLayers(
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
        (4): StackedConvLayers(
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
      )
    )
  )
  (proposal_generator): FCOS3D(
    (fcos_head): FCOSHead(
      (cls_tower): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): GroupNorm(32, 64, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (4): GroupNorm(32, 64, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (7): GroupNorm(32, 64, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (10): GroupNorm(32, 64, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (bbox_tower): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): GroupNorm(32, 64, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (4): GroupNorm(32, 64, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (7): GroupNorm(32, 64, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (10): GroupNorm(32, 64, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (share_tower): Sequential()
      (cls_logits): Conv3d(64, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bbox_pred): Conv3d(64, 6, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (ctrness): Conv3d(64, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (cplness): Conv3d(64, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
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
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (tower): Sequential(
      (0): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (ups): Sequential(
      (0): Conv3d(64, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  )
  (controller): Conv3d(64, 177, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
)
model buff size: 0.012MB, model param size:24.126MB
{'UInst3D': {'backbone': {'FPN3D': {'bottom_up': {'UNETD': {'conv_blocks_context': {'ModuleList': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 1792}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 128}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 1920}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 27712}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 128}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 27840}},
                                                                                                                                                         'size': 29760}},
                                                                                                                               'size': 29760}},
                                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 55424}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 256}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 55680}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 110720}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 256}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 110976}},
                                                                                                                                                         'size': 166656}},
                                                                                                                               'size': 166656}},
                                                                                                   '2': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 221440}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 221952}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 442624}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 512}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 443136}},
                                                                                                                                                         'size': 665088}},
                                                                                                                               'size': 665088}},
                                                                                                   '3': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 885248}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 886272}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 1769984}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 1024}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 1771008}},
                                                                                                                                                         'size': 2657280}},
                                                                                                                               'size': 2657280}},
                                                                                                   '4': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 3539968}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 3542016}},
                                                                                                                                                         '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 7078912}},
                                                                                                                                                                                         'instnorm': {'BatchNorm3d': {'size': 2048}},
                                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                                         'size': 7080960}},
                                                                                                                                                         'size': 10622976}},
                                                                                                                               'size': 10622976}},
                                                                                                   'size': 14141760}},
                                                            'size': 14141760}},
                                    'fpn_lateral1': {'Conv3d': {'size': 8448}},
                                    'fpn_lateral2': {'Conv3d': {'size': 16640}},
                                    'fpn_lateral3': {'Conv3d': {'size': 33024}},
                                    'fpn_lateral4': {'Conv3d': {'size': 65792}},
                                    'fpn_output1': {'Conv3d': {'size': 442624}},
                                    'fpn_output2': {'Conv3d': {'size': 442624}},
                                    'fpn_output3': {'Conv3d': {'size': 442624}},
                                    'fpn_output4': {'Conv3d': {'size': 442624}},
                                    'size': 16921408,
                                    'top_block': {'LastLevelP6P7': {'p6': {'Conv3d': {'size': 442624}},
                                                                    'p7': {'Conv3d': {'size': 442624}},
                                                                    'size': 885248}}}},
             'controller': {'Conv3d': {'size': 1224132}},
             'mask_branch': {'MaskBranch3D': {'refine': {'ModuleList': {'0': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 512}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442880}},
                                                                        '1': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 512}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442880}},
                                                                        '2': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 512}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442880}},
                                                                        '3': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                             '1': {'BatchNorm3d': {'size': 512}},
                                                                                             '2': {'ReLU': {'size': 0}},
                                                                                             'size': 442880}},
                                                                        'size': 1771520}},
                                              'size': 3545120,
                                              'tower': {'Sequential': {'0': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                            '1': {'BatchNorm3d': {'size': 512}},
                                                                                            '2': {'ReLU': {'size': 0}},
                                                                                            'size': 442880}},
                                                                       '1': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                            '1': {'BatchNorm3d': {'size': 512}},
                                                                                            '2': {'ReLU': {'size': 0}},
                                                                                            'size': 442880}},
                                                                       '2': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                            '1': {'BatchNorm3d': {'size': 512}},
                                                                                            '2': {'ReLU': {'size': 0}},
                                                                                            'size': 442880}},
                                                                       '3': {'Sequential': {'0': {'Conv3d': {'size': 442368}},
                                                                                            '1': {'BatchNorm3d': {'size': 512}},
                                                                                            '2': {'ReLU': {'size': 0}},
                                                                                            'size': 442880}},
                                                                       'size': 1771520}},
                                              'ups': {'Sequential': {'0': {'Conv3d': {'size': 2080}},
                                                                     'size': 2080}}}},
             'mask_head': {'DynamicMaskHead3D': {'size': 0}},
             'proposal_generator': {'FCOS3D': {'fcos_head': {'FCOSHead': {'bbox_pred': {'Conv3d': {'size': 41496}},
                                                                          'bbox_tower': {'Sequential': {'0': {'Conv3d': {'size': 442624}},
                                                                                                        '1': {'GroupNorm': {'size': 512}},
                                                                                                        '10': {'GroupNorm': {'size': 512}},
                                                                                                        '11': {'ReLU': {'size': 0}},
                                                                                                        '2': {'ReLU': {'size': 0}},
                                                                                                        '3': {'Conv3d': {'size': 442624}},
                                                                                                        '4': {'GroupNorm': {'size': 512}},
                                                                                                        '5': {'ReLU': {'size': 0}},
                                                                                                        '6': {'Conv3d': {'size': 442624}},
                                                                                                        '7': {'GroupNorm': {'size': 512}},
                                                                                                        '8': {'ReLU': {'size': 0}},
                                                                                                        '9': {'Conv3d': {'size': 442624}},
                                                                                                        'size': 1772544}},
                                                                          'cls_logits': {'Conv3d': {'size': 6916}},
                                                                          'cls_tower': {'Sequential': {'0': {'Conv3d': {'size': 442624}},
                                                                                                       '1': {'GroupNorm': {'size': 512}},
                                                                                                       '10': {'GroupNorm': {'size': 512}},
                                                                                                       '11': {'ReLU': {'size': 0}},
                                                                                                       '2': {'ReLU': {'size': 0}},
                                                                                                       '3': {'Conv3d': {'size': 442624}},
                                                                                                       '4': {'GroupNorm': {'size': 512}},
                                                                                                       '5': {'ReLU': {'size': 0}},
                                                                                                       '6': {'Conv3d': {'size': 442624}},
                                                                                                       '7': {'GroupNorm': {'size': 512}},
                                                                                                       '8': {'ReLU': {'size': 0}},
                                                                                                       '9': {'Conv3d': {'size': 442624}},
                                                                                                       'size': 1772544}},
                                                                          'cplness': {'Conv3d': {'size': 6916}},
                                                                          'ctrness': {'Conv3d': {'size': 6916}},
                                                                          'scales': {'ModuleList': {'0': {'Scale': {'size': 4}},
                                                                                                    '1': {'Scale': {'size': 4}},
                                                                                                    '2': {'Scale': {'size': 4}},
                                                                                                    '3': {'Scale': {'size': 4}},
                                                                                                    '4': {'Scale': {'size': 4}},
                                                                                                    'size': 20}},
                                                                          'share_tower': {'Sequential': {'size': 0}},
                                                                          'size': 3607352}},
                                               'fcos_outputs': {'FCOSOutputs3D': {'loc_loss_func': {'IOULoss': {'size': 0}},
                                                                                  'size': 0}},
                                               'size': 3607352}},
             'size': 25298012}}
