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
model size: 62.792MB
