import warnings, os


warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'D:\PycharmProjects\dataset\data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=0,
                project='runs/train',
                name='exp1uav',
                )