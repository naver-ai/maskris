def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDatasetTest, ReferDatasetMasking
    if image_set == 'val':
        ds = ReferDatasetTest(args,
                              split=image_set,
                              image_transforms=transform,
                              target_transforms=None)
    else:
        ds = ReferDatasetMasking(args,
                                 split=image_set,
                                 image_transforms=transform,
                                 target_transforms=None,
                                 txt_mask_ratio=args.txt_mask_ratio,
                                 txt_mask_ratio_sub=args.txt_mask_ratio_sub)

    num_classes = 2

    return ds, num_classes
