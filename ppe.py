""""
根据训练好的模型，进行推理
"""
from operator import truediv
import cv2
import numpy as np
import torch
import time

class PPE_detector:
    def __init__(self) :
        #加载模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weights/ppe_yolo_n.pt',source='local')
        self.model.conf =0.4
        #获取视频流
        self.cap = cv2.VideoCapture(0)
        

    def detect(self):
        
        #获取视频流的每一帧

        while True:

            ret， frame=self.cap.read()

            #画面翻转
            frame =cv2.flip(frame,1)

            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


            #执行推理过程
            results = self.model(frmae_rgb)
            result_np = results.pandas().xyxy[0].to_numpy()

            #绘制边界框
            for box in result_np:
                1,t,r,b = box[:4].astype('int')
                label_id =box[5]
                if label_id == 0:
                    cv2.rectangle(frmae,(1,t),(r,b),(0,255,0),5) 
                else:
                    cv2.rectangle(frmae,(1,t),(r,b),(255,0,255),5) 
                


            #显示画面
            cv2.imshow('PPE.demo',frame)
            if cv2.waitKey(10)& 0xff == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()



    
ppe = PPE_detector()
ppe.detect()


