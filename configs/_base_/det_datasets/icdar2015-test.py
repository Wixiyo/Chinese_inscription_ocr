dataset_type = 'IcdarDataset'
data_root = '../../data/ICDAR_2015'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/',
    pipeline=None)

train_list = [train]

test_list = [test]
