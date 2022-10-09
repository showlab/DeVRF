_base_ = './default.py'

expname = ''
basedir = 'path to store exp'

data = dict(
    datadir='path to dataset',
    dataset_type='blender',
    white_bkgd=True,
)

fine_train = dict(
    static_model_path = "path to the pretrained static model",
)