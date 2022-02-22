CondInst(
  (backbone): FPN(
    (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (top_block): LastLevelP6P7(
      (p6): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (p7): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (bottom_up): ResNet(
      (stem): BasicStem(
        (conv1): Conv2d(
          3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
      )
      (res2): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv1): Conv2d(
            64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
      )
      (res3): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv1): Conv2d(
            256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
      )
      (res4): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
          (conv1): Conv2d(
            512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (4): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (5): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
      )
      (res5): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
          (conv1): Conv2d(
            1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
      )
    )
  )
  (proposal_generator): FCOS(
    (fcos_head): FCOSHead(
      (cls_tower): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): GroupNorm(32, 256, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): GroupNorm(32, 256, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (10): GroupNorm(32, 256, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (bbox_tower): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): GroupNorm(32, 256, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): GroupNorm(32, 256, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (10): GroupNorm(32, 256, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (share_tower): Sequential()
      (cls_logits): Conv2d(256, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bbox_pred): Conv2d(256, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (ctrness): Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (scales): ModuleList(
        (0): Scale()
        (1): Scale()
        (2): Scale()
        (3): Scale()
        (4): Scale()
      )
    )
    (fcos_outputs): FCOSOutputs(
      (loc_loss_func): IOULoss()
    )
  )
  (mask_head): DynamicMaskHead()
  (mask_branch): MaskBranch(
    (refine): ModuleList(
      (0): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (tower): Sequential(
      (0): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (4): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (controller): Conv2d(256, 233, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)

{'CUDNN_BENCHMARK': False,
 'DATALOADER': {'ASPECT_RATIO_GROUPING': True,
                'FILTER_EMPTY_ANNOTATIONS': True,
                'NUM_WORKERS': 4,
                'REPEAT_THRESHOLD': 0.0,
                'SAMPLER_TRAIN': 'TrainingSampler'},
 'DATASETS': {'PRECOMPUTED_PROPOSAL_TOPK_TEST': 1000,
              'PRECOMPUTED_PROPOSAL_TOPK_TRAIN': 2000,
              'PROPOSAL_FILES_TEST': (),
              'PROPOSAL_FILES_TRAIN': (),
              'TEST': ('coco_2017_val',),
              'TRAIN': ('coco_2017_train',)},
 'GLOBAL': CfgNode({'HACK': 1.0}),
 'INPUT': {'CROP': {'CROP_INSTANCE': True,
                    'ENABLED': False,
                    'SIZE': [0.9, 0.9],
                    'TYPE': 'relative_range'},
           'FORMAT': 'BGR',
           'HFLIP_TRAIN': True,
           'MASK_FORMAT': 'polygon',
           'MAX_SIZE_TEST': 1333,
           'MAX_SIZE_TRAIN': 1333,
           'MIN_SIZE_TEST': 800,
           'MIN_SIZE_TRAIN': (640, 672, 704, 736, 768, 800),
           'MIN_SIZE_TRAIN_SAMPLING': 'choice',
           'RANDOM_FLIP': 'horizontal'},
 'MODEL': {'ANCHOR_GENERATOR': {'ANGLES': [[-90, 0, 90]],
                                'ASPECT_RATIOS': [[0.5, 1.0, 2.0]],
                                'NAME': 'DefaultAnchorGenerator',
                                'OFFSET': 0.0,
                                'SIZES': [[32, 64, 128, 256, 512]]},
           'BACKBONE': {'ANTI_ALIAS': False,
                        'FREEZE_AT': 2,
                        'NAME': 'build_fcos_resnet_fpn_backbone'},
           'BASIS_MODULE': {'ANN_SET': 'coco',
                            'COMMON_STRIDE': 8,
                            'CONVS_DIM': 128,
                            'IN_FEATURES': ['p3', 'p4', 'p5'],
                            'LOSS_ON': False,
                            'LOSS_WEIGHT': 0.3,
                            'NAME': 'ProtoNet',
                            'NORM': 'SyncBN',
                            'NUM_BASES': 4,
                            'NUM_CLASSES': 80,
                            'NUM_CONVS': 3},
           'BATEXT': {'CANONICAL_SIZE': 96,
                      'CONV_DIM': 256,
                      'CUSTOM_DICT': '',
                      'IN_FEATURES': ['p2', 'p3', 'p4'],
                      'NUM_CHARS': 25,
                      'NUM_CONV': 2,
                      'POOLER_RESOLUTION': (8, 32),
                      'POOLER_SCALES': (0.25, 0.125, 0.0625),
                      'RECOGNITION_LOSS': 'ctc',
                      'RECOGNIZER': 'attn',
                      'SAMPLING_RATIO': 1,
                      'USE_AET': False,
                      'USE_COORDCONV': False,
                      'VOC_SIZE': 96},
           'BLENDMASK': {'ATTN_SIZE': 14,
                         'BOTTOM_RESOLUTION': 56,
                         'INSTANCE_LOSS_WEIGHT': 1.0,
                         'POOLER_SAMPLING_RATIO': 1,
                         'POOLER_SCALES': (0.25,),
                         'POOLER_TYPE': 'ROIAlignV2',
                         'TOP_INTERP': 'bilinear',
                         'VISUALIZE': False},
           'BOXINST': {'BOTTOM_PIXELS_REMOVED': 10,
                       'ENABLED': True,
                       'PAIRWISE': {'COLOR_THRESH': 0.3,
                                    'DILATION': 2,
                                    'SIZE': 3,
                                    'WARMUP_ITERS': 10000}},
           'BiFPN': {'IN_FEATURES': ['res2', 'res3', 'res4', 'res5'],
                     'NORM': '',
                     'NUM_REPEATS': 6,
                     'OUT_CHANNELS': 160},
           'CONDINST': {'BOTTOM_PIXELS_REMOVED': -1,
                        'MASK_BRANCH': {'CHANNELS': 128,
                                        'IN_FEATURES': ['p3', 'p4', 'p5'],
                                        'NORM': 'BN',
                                        'NUM_CONVS': 4,
                                        'OUT_CHANNELS': 16,
                                        'SEMANTIC_LOSS_ON': False},
                        'MASK_HEAD': {'CHANNELS': 8,
                                      'DISABLE_REL_COORDS': False,
                                      'NUM_LAYERS': 3,
                                      'USE_FP16': False},
                        'MASK_OUT_STRIDE': 4,
                        'MAX_PROPOSALS': -1,
                        'TOPK_PROPOSALS_PER_IM': 64},
           'DEVICE': 'cuda',
           'DLA': {'CONV_BODY': 'DLA34',
                   'NORM': 'FrozenBN',
                   'OUT_FEATURES': ['stage2', 'stage3', 'stage4', 'stage5']},
           'FCOS': {'BOX_QUALITY': 'ctrness',
                    'CENTER_SAMPLE': True,
                    'FPN_STRIDES': [8, 16, 32, 64, 128],
                    'INFERENCE_TH_TEST': 0.3,
                    'INFERENCE_TH_TRAIN': 0.05,
                    'IN_FEATURES': ['p3', 'p4', 'p5', 'p6', 'p7'],
                    'LOC_LOSS_TYPE': 'giou',
                    'LOSS_ALPHA': 0.25,
                    'LOSS_GAMMA': 2.0,
                    'LOSS_NORMALIZER_CLS': 'fg',
                    'LOSS_WEIGHT_CLS': 1.0,
                    'NMS_TH': 0.6,
                    'NORM': 'GN',
                    'NUM_BOX_CONVS': 4,
                    'NUM_CLASSES': 80,
                    'NUM_CLS_CONVS': 4,
                    'NUM_SHARE_CONVS': 0,
                    'POST_NMS_TOPK_TEST': 100,
                    'POST_NMS_TOPK_TRAIN': 100,
                    'POS_RADIUS': 1.5,
                    'PRE_NMS_TOPK_TEST': 1000,
                    'PRE_NMS_TOPK_TRAIN': 1000,
                    'PRIOR_PROB': 0.01,
                    'SIZES_OF_INTEREST': [64, 128, 256, 512],
                    'THRESH_WITH_CTR': True,
                    'TOP_LEVELS': 2,
                    'USE_DEFORMABLE': False,
                    'USE_RELU': True,
                    'USE_SCALE': True,
                    'YIELD_BOX_FEATURES': False,
                    'YIELD_PROPOSAL': False},
           'FCPOSE': {'ATTN_LEN': 2737,
                      'BASIS_MODULE': {'BN_TYPE': 'SyncBN',
                                       'COMMON_STRIDE': 8,
                                       'CONVS_DIM': 128,
                                       'LOSS_WEIGHT': 0.2,
                                       'NUM_BASES': 32,
                                       'NUM_CLASSES': 17},
                      'DISTANCE_NORM': 12.0,
                      'DYNAMIC_CHANNELS': 32,
                      'FOCAL_LOSS_ALPHA': 0.25,
                      'FOCAL_LOSS_GAMMA': 2.0,
                      'GT_HEATMAP_STRIDE': 2,
                      'HEAD_HEATMAP_SIGMA': 0.01,
                      'HEATMAP_SIGMA': 1.8,
                      'LOSS_WEIGHT_DIRECTION': 9.0,
                      'LOSS_WEIGHT_KEYPOINT': 2.5,
                      'MAX_PROPOSALS': 70,
                      'PROPOSALS_PER_INST': 70,
                      'SIGMA': 1},
           'FCPOSE_ON': False,
           'FPN': {'FUSE_TYPE': 'sum',
                   'IN_FEATURES': ['res3', 'res4', 'res5'],
                   'NORM': '',
                   'OUT_CHANNELS': 256},
           'KEYPOINT_ON': False,
           'LOAD_PROPOSALS': False,
           'MASK_ON': True,
           'MEInst': {'AGNOSTIC': True,
                      'CENTER_SAMPLE': True,
                      'DIM_MASK': 60,
                      'FLAG_PARAMETERS': False,
                      'FPN_STRIDES': [8, 16, 32, 64, 128],
                      'GCN_KERNEL_SIZE': 9,
                      'INFERENCE_TH_TEST': 0.3,
                      'INFERENCE_TH_TRAIN': 0.05,
                      'IN_FEATURES': ['p3', 'p4', 'p5', 'p6', 'p7'],
                      'IOU_LABELS': [0, 1],
                      'IOU_THRESHOLDS': [0.5],
                      'LAST_DEFORMABLE': False,
                      'LOC_LOSS_TYPE': 'giou',
                      'LOSS_ALPHA': 0.25,
                      'LOSS_GAMMA': 2.0,
                      'LOSS_ON_MASK': False,
                      'MASK_LOSS_TYPE': 'mse',
                      'MASK_ON': True,
                      'MASK_SIZE': 28,
                      'NMS_TH': 0.6,
                      'NORM': 'GN',
                      'NUM_BOX_CONVS': 4,
                      'NUM_CLASSES': 80,
                      'NUM_CLS_CONVS': 4,
                      'NUM_MASK_CONVS': 4,
                      'NUM_SHARE_CONVS': 0,
                      'PATH_COMPONENTS': 'datasets/coco/components/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz',
                      'POST_NMS_TOPK_TEST': 100,
                      'POST_NMS_TOPK_TRAIN': 100,
                      'POS_RADIUS': 1.5,
                      'PRE_NMS_TOPK_TEST': 1000,
                      'PRE_NMS_TOPK_TRAIN': 1000,
                      'PRIOR_PROB': 0.01,
                      'SIGMOID': True,
                      'SIZES_OF_INTEREST': [64, 128, 256, 512],
                      'THRESH_WITH_CTR': False,
                      'TOP_LEVELS': 2,
                      'TYPE_DEFORMABLE': 'DCNv1',
                      'USE_DEFORMABLE': False,
                      'USE_GCN_IN_MASK': False,
                      'USE_RELU': True,
                      'USE_SCALE': True,
                      'WHITEN': True},
           'META_ARCHITECTURE': 'CondInst',
           'MOBILENET': False,
           'PANOPTIC_FPN': {'COMBINE': {'ENABLED': True,
                                        'INSTANCES_CONFIDENCE_THRESH': 0.3,
                                        'OVERLAP_THRESH': 0.5,
                                        'STUFF_AREA_LIMIT': 4096},
                            'INSTANCE_LOSS_WEIGHT': 1.0},
           'PIXEL_MEAN': [103.53, 116.28, 123.675],
           'PIXEL_STD': [1.0, 1.0, 1.0],
           'PROPOSAL_GENERATOR': CfgNode({'NAME': 'FCOS', 'MIN_SIZE': 0}),
           'RESNETS': {'DEFORM_INTERVAL': 1,
                       'DEFORM_MODULATED': False,
                       'DEFORM_NUM_GROUPS': 1,
                       'DEFORM_ON_PER_STAGE': [False, False, False, False],
                       'DEPTH': 50,
                       'NORM': 'FrozenBN',
                       'NUM_GROUPS': 1,
                       'OUT_FEATURES': ['res3', 'res4', 'res5'],
                       'RES2_OUT_CHANNELS': 256,
                       'RES5_DILATION': 1,
                       'STEM_OUT_CHANNELS': 64,
                       'STRIDE_IN_1X1': True,
                       'WIDTH_PER_GROUP': 64},
           'RETINANET': {'BBOX_REG_LOSS_TYPE': 'smooth_l1',
                         'BBOX_REG_WEIGHTS': (1.0, 1.0, 1.0, 1.0),
                         'FOCAL_LOSS_ALPHA': 0.25,
                         'FOCAL_LOSS_GAMMA': 2.0,
                         'IN_FEATURES': ['p3', 'p4', 'p5', 'p6', 'p7'],
                         'IOU_LABELS': [0, -1, 1],
                         'IOU_THRESHOLDS': [0.4, 0.5],
                         'NMS_THRESH_TEST': 0.5,
                         'NORM': '',
                         'NUM_CLASSES': 80,
                         'NUM_CONVS': 4,
                         'PRIOR_PROB': 0.01,
                         'SCORE_THRESH_TEST': 0.3,
                         'SMOOTH_L1_LOSS_BETA': 0.1,
                         'TOPK_CANDIDATES_TEST': 1000},
           'ROI_BOX_CASCADE_HEAD': {'BBOX_REG_WEIGHTS': ((10.0, 10.0, 5.0, 5.0),
                                                         (20.0,
                                                          20.0,
                                                          10.0,
                                                          10.0),
                                                         (30.0,
                                                          30.0,
                                                          15.0,
                                                          15.0)),
                                    'IOUS': (0.5, 0.6, 0.7)},
           'ROI_BOX_HEAD': {'BBOX_REG_LOSS_TYPE': 'smooth_l1',
                            'BBOX_REG_LOSS_WEIGHT': 1.0,
                            'BBOX_REG_WEIGHTS': (10.0, 10.0, 5.0, 5.0),
                            'CLS_AGNOSTIC_BBOX_REG': False,
                            'CONV_DIM': 256,
                            'FC_DIM': 1024,
                            'NAME': '',
                            'NORM': '',
                            'NUM_CONV': 0,
                            'NUM_FC': 0,
                            'POOLER_RESOLUTION': 14,
                            'POOLER_SAMPLING_RATIO': 0,
                            'POOLER_TYPE': 'ROIAlignV2',
                            'SMOOTH_L1_BETA': 0.0,
                            'TRAIN_ON_PRED_BOXES': False},
           'ROI_HEADS': {'BATCH_SIZE_PER_IMAGE': 512,
                         'IN_FEATURES': ['res4'],
                         'IOU_LABELS': [0, 1],
                         'IOU_THRESHOLDS': [0.5],
                         'NAME': 'Res5ROIHeads',
                         'NMS_THRESH_TEST': 0.5,
                         'NUM_CLASSES': 80,
                         'POSITIVE_FRACTION': 0.25,
                         'PROPOSAL_APPEND_GT': True,
                         'SCORE_THRESH_TEST': 0.3},
           'ROI_KEYPOINT_HEAD': {'CONV_DIMS': (512,
                                               512,
                                               512,
                                               512,
                                               512,
                                               512,
                                               512,
                                               512),
                                 'LOSS_WEIGHT': 1.0,
                                 'MIN_KEYPOINTS_PER_IMAGE': 1,
                                 'NAME': 'KRCNNConvDeconvUpsampleHead',
                                 'NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS': True,
                                 'NUM_KEYPOINTS': 17,
                                 'POOLER_RESOLUTION': 14,
                                 'POOLER_SAMPLING_RATIO': 0,
                                 'POOLER_TYPE': 'ROIAlignV2'},
           'ROI_MASK_HEAD': {'CLS_AGNOSTIC_MASK': False,
                             'CONV_DIM': 256,
                             'NAME': 'MaskRCNNConvUpsampleHead',
                             'NORM': '',
                             'NUM_CONV': 0,
                             'POOLER_RESOLUTION': 14,
                             'POOLER_SAMPLING_RATIO': 0,
                             'POOLER_TYPE': 'ROIAlignV2'},
           'RPN': {'BATCH_SIZE_PER_IMAGE': 256,
                   'BBOX_REG_LOSS_TYPE': 'smooth_l1',
                   'BBOX_REG_LOSS_WEIGHT': 1.0,
                   'BBOX_REG_WEIGHTS': (1.0, 1.0, 1.0, 1.0),
                   'BOUNDARY_THRESH': -1,
                   'CONV_DIMS': [-1],
                   'HEAD_NAME': 'StandardRPNHead',
                   'IN_FEATURES': ['res4'],
                   'IOU_LABELS': [0, -1, 1],
                   'IOU_THRESHOLDS': [0.3, 0.7],
                   'LOSS_WEIGHT': 1.0,
                   'NMS_THRESH': 0.7,
                   'POSITIVE_FRACTION': 0.5,
                   'POST_NMS_TOPK_TEST': 1000,
                   'POST_NMS_TOPK_TRAIN': 2000,
                   'PRE_NMS_TOPK_TEST': 6000,
                   'PRE_NMS_TOPK_TRAIN': 12000,
                   'SMOOTH_L1_BETA': 0.0},
           'SEM_SEG_HEAD': {'COMMON_STRIDE': 4,
                            'CONVS_DIM': 128,
                            'IGNORE_VALUE': 255,
                            'IN_FEATURES': ['p2', 'p3', 'p4', 'p5'],
                            'LOSS_WEIGHT': 1.0,
                            'NAME': 'SemSegFPNHead',
                            'NORM': 'GN',
                            'NUM_CLASSES': 54},
           'SOLOV2': {'FPN_INSTANCE_STRIDES': [8, 8, 16, 32, 32],
                      'FPN_SCALE_RANGES': ((1, 96),
                                           (48, 192),
                                           (96, 384),
                                           (192, 768),
                                           (384, 2048)),
                      'INSTANCE_CHANNELS': 512,
                      'INSTANCE_IN_CHANNELS': 256,
                      'INSTANCE_IN_FEATURES': ['p2', 'p3', 'p4', 'p5', 'p6'],
                      'LOSS': {'DICE_WEIGHT': 3.0,
                               'FOCAL_ALPHA': 0.25,
                               'FOCAL_GAMMA': 2.0,
                               'FOCAL_USE_SIGMOID': True,
                               'FOCAL_WEIGHT': 1.0},
                      'MASK_CHANNELS': 128,
                      'MASK_IN_CHANNELS': 256,
                      'MASK_IN_FEATURES': ['p2', 'p3', 'p4', 'p5'],
                      'MASK_THR': 0.5,
                      'MAX_PER_IMG': 100,
                      'NMS_KERNEL': 'gaussian',
                      'NMS_PRE': 500,
                      'NMS_SIGMA': 2,
                      'NMS_TYPE': 'matrix',
                      'NORM': 'GN',
                      'NUM_CLASSES': 80,
                      'NUM_GRIDS': [40, 36, 24, 16, 12],
                      'NUM_INSTANCE_CONVS': 4,
                      'NUM_KERNELS': 256,
                      'NUM_MASKS': 256,
                      'PRIOR_PROB': 0.01,
                      'SCORE_THR': 0.1,
                      'SIGMA': 0.2,
                      'TYPE_DCN': 'DCN',
                      'UPDATE_THR': 0.05,
                      'USE_COORD_CONV': True,
                      'USE_DCN_IN_INSTANCE': False},
           'TOP_MODULE': CfgNode({'NAME': 'conv', 'DIM': 16}),
           'VOVNET': {'BACKBONE_OUT_CHANNELS': 256,
                      'CONV_BODY': 'V-39-eSE',
                      'NORM': 'FrozenBN',
                      'OUT_CHANNELS': 256,
                      'OUT_FEATURES': ['stage2', 'stage3', 'stage4', 'stage5']},
           'WEIGHTS': 'BoxInst_MS_R_50_3x.pth'},
 'OUTPUT_DIR': 'output/boxinst_MS_R_50_3x',
 'SEED': -1,
 'SOLVER': {'AMP': CfgNode({'ENABLED': False}),
            'BASE_LR': 0.01,
            'BIAS_LR_FACTOR': 1.0,
            'CHECKPOINT_PERIOD': 5000,
            'CLIP_GRADIENTS': {'CLIP_TYPE': 'value',
                               'CLIP_VALUE': 1.0,
                               'ENABLED': False,
                               'NORM_TYPE': 2.0},
            'GAMMA': 0.1,
            'IMS_PER_BATCH': 16,
            'LR_SCHEDULER_NAME': 'WarmupMultiStepLR',
            'MAX_ITER': 270000,
            'MOMENTUM': 0.9,
            'NESTEROV': False,
            'REFERENCE_WORLD_SIZE': 0,
            'STEPS': (210000, 250000),
            'WARMUP_FACTOR': 0.001,
            'WARMUP_ITERS': 1000,
            'WARMUP_METHOD': 'linear',
            'WEIGHT_DECAY': 0.0001,
            'WEIGHT_DECAY_BIAS': None,
            'WEIGHT_DECAY_NORM': 0.0},
 'TEST': {'AUG': {'ENABLED': False,
                  'FLIP': True,
                  'MAX_SIZE': 4000,
                  'MIN_SIZES': (400,
                                500,
                                600,
                                700,
                                800,
                                900,
                                1000,
                                1100,
                                1200)},
          'DETECTIONS_PER_IMAGE': 100,
          'EVAL_PERIOD': 0,
          'EXPECTED_RESULTS': [],
          'KEYPOINT_OKS_SIGMAS': [],
          'PRECISE_BN': CfgNode({'ENABLED': False, 'NUM_ITER': 200})},
 'VERSION': 2,
 'VIS_PERIOD': 0}
