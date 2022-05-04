dataset_type = 'OCRDataset'

root = '../../../data'
img_prefix = f'{root}/imgs'
train_anno_file1 = f'{root}/train.txt'

train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=100,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train_anno_file2 = f'{root}/train.txt'
train2 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=100,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

test_anno_file1 = f'{root}/test.txt'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

train_list = [train1, train2]

test_list = [test]
