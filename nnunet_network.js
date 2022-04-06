Generic_UNet(
  (conv_blocks_localization): ModuleList(
    (0): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(640, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(320, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (1): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (2): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (3): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (4): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
  )
  (conv_blocks_context): ModuleList(
    (0): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (1): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (2): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (3): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (4): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv3d(256, 320, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv3d(320, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (instnorm): InstanceNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (5): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(320, 320, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv3d(320, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (instnorm): InstanceNorm3d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
  )
  (td): ModuleList()
  (tu): ModuleList(
    (0): ConvTranspose3d(320, 320, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    (1): ConvTranspose3d(320, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    (2): ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    (3): ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    (4): ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
  )
  (seg_outputs): ModuleList(
    (0): Conv3d(320, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    (1): Conv3d(256, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    (2): Conv3d(128, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    (3): Conv3d(64, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    (4): Conv3d(32, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
  )
)
model buff size: 0.000MB, model param size:119.005MB
{'Generic_UNet': {'conv_blocks_context': {'ModuleList': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 3584}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 256}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 3840}},
                                                                                                               '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 110720}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 256}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 110976}},
                                                                                                               'size': 114816}},
                                                                                     'size': 114816}},
                                                         '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 221440}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 512}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 221952}},
                                                                                                               '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 442624}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 512}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 443136}},
                                                                                                               'size': 665088}},
                                                                                     'size': 665088}},
                                                         '2': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 885248}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 1024}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 886272}},
                                                                                                               '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 1769984}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 1024}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 1771008}},
                                                                                                               'size': 2657280}},
                                                                                     'size': 2657280}},
                                                         '3': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 3539968}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 2048}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 3542016}},
                                                                                                               '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 7078912}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 2048}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 7080960}},
                                                                                                               'size': 10622976}},
                                                                                     'size': 10622976}},
                                                         '4': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 8848640}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 2560}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 8851200}},
                                                                                                               '1': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 11060480}},
                                                                                                                                               'instnorm': {'InstanceNorm3d': {'size': 2560}},
                                                                                                                                               'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                               'size': 11063040}},
                                                                                                               'size': 19914240}},
                                                                                     'size': 19914240}},
                                                         '5': {'Sequential': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 11060480}},
                                                                                                                                                                    'instnorm': {'InstanceNorm3d': {'size': 2560}},
                                                                                                                                                                    'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                    'size': 11063040}},
                                                                                                                                    'size': 11063040}},
                                                                                                          'size': 11063040}},
                                                                              '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 11060480}},
                                                                                                                                                                    'instnorm': {'InstanceNorm3d': {'size': 2560}},
                                                                                                                                                                    'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                    'size': 11063040}},
                                                                                                                                    'size': 11063040}},
                                                                                                          'size': 11063040}},
                                                                              'size': 22126080}},
                                                         'size': 56100480}},
                  'conv_blocks_localization': {'ModuleList': {'0': {'Sequential': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 22119680}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 2560}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 22122240}},
                                                                                                                                         'size': 22122240}},
                                                                                                               'size': 22122240}},
                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 11060480}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 2560}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 11063040}},
                                                                                                                                         'size': 11063040}},
                                                                                                               'size': 11063040}},
                                                                                   'size': 33185280}},
                                                              '1': {'Sequential': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 14156800}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 2048}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 14158848}},
                                                                                                                                         'size': 14158848}},
                                                                                                               'size': 14158848}},
                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 7078912}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 2048}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 7080960}},
                                                                                                                                         'size': 7080960}},
                                                                                                               'size': 7080960}},
                                                                                   'size': 21239808}},
                                                              '2': {'Sequential': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 3539456}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 1024}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 3540480}},
                                                                                                                                         'size': 3540480}},
                                                                                                               'size': 3540480}},
                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 1769984}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 1024}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 1771008}},
                                                                                                                                         'size': 1771008}},
                                                                                                               'size': 1771008}},
                                                                                   'size': 5311488}},
                                                              '3': {'Sequential': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 884992}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 512}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 885504}},
                                                                                                                                         'size': 885504}},
                                                                                                               'size': 885504}},
                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 442624}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 512}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 443136}},
                                                                                                                                         'size': 443136}},
                                                                                                               'size': 443136}},
                                                                                   'size': 1328640}},
                                                              '4': {'Sequential': {'0': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 221312}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 256}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 221568}},
                                                                                                                                         'size': 221568}},
                                                                                                               'size': 221568}},
                                                                                   '1': {'StackedConvLayers': {'blocks': {'Sequential': {'0': {'ConvDropoutNormNonlin': {'conv': {'Conv3d': {'size': 110720}},
                                                                                                                                                                         'instnorm': {'InstanceNorm3d': {'size': 256}},
                                                                                                                                                                         'lrelu': {'LeakyReLU': {'size': 0}},
                                                                                                                                                                         'size': 110976}},
                                                                                                                                         'size': 110976}},
                                                                                                               'size': 110976}},
                                                                                   'size': 332544}},
                                                              'size': 61397760}},
                  'seg_outputs': {'ModuleList': {'0': {'Conv3d': {'size': 5120}},
                                                 '1': {'Conv3d': {'size': 4096}},
                                                 '2': {'Conv3d': {'size': 2048}},
                                                 '3': {'Conv3d': {'size': 1024}},
                                                 '4': {'Conv3d': {'size': 512}},
                                                 'size': 12800}},
                  'size': 124785536,
                  'td': {'ModuleList': {'size': 0}},
                  'tu': {'ModuleList': {'0': {'ConvTranspose3d': {'size': 3276800}},
                                        '1': {'ConvTranspose3d': {'size': 2621440}},
                                        '2': {'ConvTranspose3d': {'size': 1048576}},
                                        '3': {'ConvTranspose3d': {'size': 262144}},
                                        '4': {'ConvTranspose3d': {'size': 65536}},
                                        'size': 7274496}}}}
