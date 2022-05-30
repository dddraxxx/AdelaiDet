from adet.utils.dataset_3d import *
from adet.utils.gridding import GriddingReverse


def add_bitmasks_from_boxes_3d(per_im_gt_inst, images, image_masks, im_s, im_h, im_w):
    # print(images.shape)
    stride = 1
    start = int(stride // 2)

    assert images.size(2) % stride == 0
    assert images.size(3) % stride == 0

    # for im_i, per_im_gt_inst in enumerate(instances):
    # images_lab = downsampled_images[im_i][None]

    per_im_boxes = per_im_gt_inst.gt_boxes.tensor
    for per_box in per_im_boxes:
        bitmask_full = torch.zeros((im_s, im_h, im_w)).float()
        bitmask_full[
            int(per_box[0]) : int(per_box[3] + 1),
            int(per_box[1]) : int(per_box[4] + 1),
            int(per_box[2]) : int(per_box[5] + 1),
        ] = 1.0

        bitmask = bitmask_full[start::stride, start::stride]

        assert bitmask.size(0) * stride == im_s
        assert bitmask.size(1) * stride == im_h
        assert bitmask.size(2) * stride == im_w

    return bitmask


def get_lits_pair(idx=2):
    """
    gt: 1, S, H, W"""
    template_idx = idx
    dset = Volumes(0, "/mnt/sdb/nnUNet/Task029_LITS/uinst_stage0/train_{}.npy")
    img, gt = dset.get_data(template_idx, read_gt=True, pt=False)
    print("gt has label ", gt.unique())
    # print((gt>0).nonzero())
    return img, (gt > 0)[:, 50 : 194 - 50].float()  # too large size bad


def get_kidney_pair(idx=2):
    template_idx = idx
    dset = Volumes(
        0,
        "/mnt/sdb/nnUNet/Task361_KiDsOnly/nnUNetData_plans_v2.1_stage0/case_{:05d}.npy",
    )
    img, gt = dset.get_data(template_idx, read_gt=True, pt=False)
    print("gt has label ", gt.unique())
    # print((gt>0).nonzero())
    return (
        img,
        (gt[:, 20:-30, : 242 // 2, -242 // 2 :] == 1).float(),
    )  # too large size bad


if __name__ == "__main__":

    # template_idx = 2
    # dset = Volumes(0)
    # template = dset.__getitem__(template_idx)

    # original_images = template['image']
    # gt_instances = template["instances"]
    # original_image_masks = torch.ones_like(original_images[0], dtype=torch.float32)
    # print(original_images.shape)
    # bitmask = add_bitmasks_from_boxes_3d(
    #     gt_instances, original_images, original_image_masks,
    #     original_images.size(-3),
    #     original_images.size(-2), original_images.size(-1)
    # )[None]
    # gt = bitmask
    item = "kidney"
    d, gt = get_lits_pair(1)
    d, gt = get_kidney_pair(2)
    from adet.utils.visualize_niigz import *

    visulize_3d(gt[0], save_name="ch1.png")
    model = GriddingReverse(max(gt.shape[-3:])).cuda()
    pc = model(gt.cuda().contiguous())[0]

    def test_scale(scale, bitmask):
        model = GriddingReverse(scale).cuda()
        pc = model(bitmask.unsqueeze(0).cuda().contiguous())
        return pc

    print(gt.shape, pc.shape)
    # exit(1)
    pc = pc[(pc != 0).any(dim=1)]
    print(pc.shape)
    save_path = f"./{item}"
    with open(save_path, "w") as f:
        for line in pc:
            f.write("{} {} {}\n".format(float(line[0]), float(line[1]), float(line[2])))

    header = (
        "ply "
        "format ascii 1.0"
        "comment made by anonymous"
        "element vertex {}"
        "property float32 x"
        "property float32 y"
        "property float32 z"
        "end_header\n"
    ).format(len(pc))
    with open(f"./{item}.ply", "w") as f:
        f.write(header)
        for line in pc:
            f.write("{} {} {}\n".format(float(line[0]), float(line[1]), float(line[2])))
