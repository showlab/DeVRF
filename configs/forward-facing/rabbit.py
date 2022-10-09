_base_ = './real_default_rabbit.py'

expname = ''
basedir = 'path to store exp'

data = dict(
    white_bkgd=True,
    datadir='path to dataset',
)

fine_train = dict(
    static_model_path = "path to the pretrained static model",
)
