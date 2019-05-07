import numpy as np
import cv2 as cv
from detection import Detection
from utils import bounding_box_naive
import math
class Track(Detection):
    def __init__(self,method,_id,det):
         
        self.conf = 0.3
        self.xmin = det.xmin
        self.ymin = det.ymin
        self.xmax = det.xmax
        self.ymax = det.ymax
        self.pred_xmin = det.xmin
        self.pred_ymin = det.ymin
        self.pred_xmax = det.xmax
        self.pred_ymax = det.ymax
        self.tracked_count = 1
        self.offset = np.array([0,0],np.float32)
        self.missed_count=0
        self.matched=True
        self.descriptor = np.array(np.zeros(det.descriptor.shape[0],np.float32))
        self.descriptor[:] = det.descriptor[:]
        self.track_id = _id
        self.method = method
        self.feature_params=dict(maxCorners=30,qualityLevel=0.3,minDistance=7,blockSize=7)
        self.lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,0.03))
        if(method=='kalman_corners'):
            self.init_kalman_tracker()
        self.init_offset_tracker()
            
    def init_offset_tracker(self):
        self.offset_tracker= cv.KalmanFilter(4,2)
        self.offset_tracker.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

        self.offset_tracker.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

        self.offset_tracker.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.001
        self.offset_tracker.correct(self.offset)
        self.offset_tracker.predict();
        self.offset_tracker.correct(self.offset)
        self.offset_tracker.predict();
        self.offset_tracker.correct(self.offset)
        self.offset_tracker.predict();
        self.offset_tracker.correct(self.offset)
        self.offset_tracker.predict();
    def init_kalman_tracker(self):
        self.kalman_tracker = cv.KalmanFilter(8,4)
        self.kalman_tracker.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]],np.float32)

        self.kalman_tracker.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1]
                                           ,[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],np.float32)

        self.kalman_tracker.processNoiseCov = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]
                                          ,[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],np.float32)*0.001

        self.kalman_tracker.predict();

        self.kalman_tracker.correct(self.corners())
        self.kalman_tracker.predict();
        self.kalman_tracker.correct(self.corners())
        self.kalman_tracker.predict();
        self.kalman_tracker.correct(self.corners())
        self.kalman_tracker.predict();
        self.kalman_tracker.correct(self.corners())
    
    def copy(self):
        other = Track(self.track_id,np.array([0,self.conf,self.xmin,self.ymin,self.xmax,self.ymax],np.float32),self.descriptor)
        other.tracked_count = self.tracked_count
        other.missed_count = self.missed_count
        other.matched = self.matched
        
        return other
    
    def update(self,det,frame_gray,prev_frame_gray):
        self.predict(prev_frame_gray,frame_gray)
        self.xmin = det.xmin
        self.ymin = det.ymin
        self.xmax=det.xmax
        self.ymax = det.ymax
        if(det.descriptor.shape[0]!=self.descriptor.shape[0]):
            self.descriptor = []
            self.descriptor = np.zeros(det.descriptor.shape[0],np.float32)
        self.descriptor[:] = det.descriptor
        self.matched = True
        self.missed_count = 0
        self.tracked_count +=1
        
        if(self.tracked_count>3):
            self.conf = det.conf
        if(self.method=='kalman_corners'):
            self.kalman_tracker.correct(self.corners())
    def apply_prediction(self,frame_gray,prev_frame_gray):
        
        self.predict(prev_frame_gray,frame_gray)
        self.xmin = self.pred_xmin
        self.ymin = self.pred_ymin
        self.xmax = self.pred_xmax
        self.ymax = self.pred_ymax
      
    def shiftKeyPointsFlow(self,frame,prev_frame):
	
        frame_grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_width = frame_grey.shape[1]
        frame_height = frame_grey.shape[0]
        prev_frame_grey = cv.cvtColor(prev_frame,cv.COLOR_BGR2GRAY)
        mask = np.zeros(frame_grey.shape, dtype = "uint8")
        cv.rectangle(mask, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), (255, 255, 255), -1)
        p0 = cv.goodFeaturesToTrack(prev_frame_grey, mask = mask, **self.feature_params)
            
        if(not p0 is None ):
            p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame_grey, frame_grey,p0, None, **self.lk_params)
            old_box = bounding_box_naive(p0)
            new_box = bounding_box_naive(p1)
            offset = [new_box[0]-old_box[0],new_box[1]-old_box[1]]
            
            new_center = self.center() +offset
            old_width = self.xmax - self.xmin
            old_height = self.ymax-self.ymin
            
            new_width = old_width * (new_box[2]/old_box[2])

            new_height = old_height * (new_box[3]/old_box[3])
            scale_change= np.max([old_width/new_width,old_height/new_height])
            if(new_width==0 or new_width>frame_width or math.isnan(new_width)):
                new_width=0
            if(new_height==0 or new_height>frame_height or math.isnan(new_height)):
                new_height=0
            self.offset_tracker.correct(np.array([offset[0],scale_change],np.float32))
            pred_offset= self.offset_tracker.predict()
            if(not math.isnan(pred_offset[0][0])):
                self.offset[0]= pred_offset[0][0]
            if(not math.isnan(pred_offset[1][0])):
                self.offset[1]= pred_offset[1][0]
            
            self.pred_xmin = new_center[0] - (new_width/2)
            self.pred_ymin=new_center[1] - (new_height/2)
            self.pred_xmax=new_center[0] + (new_width/2)
            self.pred_ymax=new_center[1]+ (new_height/2)
            
        elif(self.missed_count>0):
            self.conf=0
            
        else:
            print('2.2 no points to track')
    def predict(self,prev_frame_gray,frame_gray):
        if(self.method=='kalman_corners'):
            pred = self.kalman_tracker.predict()
            self.pred_xmin = pred[0][0]
            self.pred_ymin = pred[1][0]
            self.pred_xmax = pred[2][0]
            self.pred_ymax = pred[3][0]
            
        elif(self.method =='keypoint_flow'):
            self.shiftKeyPointsFlow(frame_gray,prev_frame_gray)
           
            
    