import numpy as np
import math
import scipy.interpolate as interp
from scipy.spatial import distance
import imutils
import time
from sklearn import preprocessing
import cv2 as cv
import time
from sklearn.utils.linear_assignment_ import linear_assignment
from utils import iou
from utils import get_distance
from utils import bounding_box_naive
from detection import Detection
from track import Track
class Tracker(object):
    def __init__(self,method='keypoint_flow'):
        self.tracking_method = method
        self.tracks = []
        self.cur_id=1
        self.frameCount =1
        
    def get_distance_matrix(self,dets):
        dists = np.zeros((len(dets),len(self.tracks)),np.float32)
        for itrack in range(len(self.tracks)):
            for ipred in range(len(dets)):
                #iou_dist = (1- iou(dets[ipred].corners(),self.tracks[itrack].corners()))
                
                #desc_dist = get_distance(self.tracks[itrack].descriptor,dets[ipred].descriptor)
                iou_overlap = iou(dets[ipred].corners(),self.tracks[itrack].corners())
                #if(iou_dist==1):
                    #iou_dist=3
               # if(iou_dist<0.7):
                    #iou_dist = 0
                dists[ipred,itrack] = -iou_overlap
        return dists
    def track(self,dets,frame_gray,prev_frame_gray):
        
        dists = self.get_distance_matrix(dets)
        matched_indices = linear_assignment(dists)
        for m in matched_indices:
            #descr_t = self.tracks[m[1]].descriptor
            #descr_p = dets[m[0]].descriptor

            if(self.tracks[m[1]].missed_count<3):
                iou_threshold=0.5
            elif(self.tracks[m[1]].missed_count<8):
                iou_threshold=0.2
            else:
                iou_threshold=0.1
            if(dists[m[0],m[1]]>iou_threshold):#iou_threshold):
                m[0] = -1
                m[1]=-1
        for trk in self.tracks:
            trk.matched=False
            
        for d,det in enumerate(dets):
            if(d not in matched_indices[:,0] ):
                
                
               
                self.tracks.append(Track(self.tracking_method,self.cur_id,det))
                
                self.cur_id+=1
                
            else:
                index = np.where(matched_indices[:,0]==d)
                index = matched_indices[index][0][1]
                self.tracks[index].update(det,frame_gray,prev_frame_gray)
        
        for t,trk in enumerate(self.tracks):
            if(t not in matched_indices[:,1] and trk.matched==False):
                trk.missed_count+=1
                if(trk.tracked_count<10):
                    if(trk.missed_count>4):
                        trk.conf = 0.3 #hide
                    elif(trk.missed_count>6):
                        trk.conf = 0.2 #remove
                elif(trk.tracked_count<30):
                    if(trk.missed_count>8):
                        trk.conf=0.3 #hide
                    elif(trk.missed_count>12):
                        trk.conf = 0.2 #remove
                else:
                    if(trk.missed_count>15):
                        trk.conf=0.3 #hide
                    elif(trk.missed_count>20):
                        trk.conf = 0.2 #remove
                if(trk.missed_count>30):
                    trk.conf=0 #remove
                
                trk.apply_prediction(frame_gray,prev_frame_gray)
       
        self.frameCount+=1
    def get_display_tracks(self):
        return [track for track in self.tracks if track.conf>0.2]
    def get_collision_points(self):
        cols = []
        objs = self.get_display_tracks()
        for obj in objs:
            if(self.point_in_collison(obj.botleft())):
                cols.append(obj.botleft())
            elif(self.point_in_collison(obj.botright())):
                cols.append(obj.botright())
            elif(self.point_in_box(self.A,obj)):
                cols.append(np.array([self.A[0],obj.ymax]))
           
            
        return cols
    def point_in_collison(self,p):
        
        if(p[1]<self.frame_height/2):
            return False
        
        q1 = (5*self.frame_height *p[0])/self.frame_width + 3*p[1] -4*self.frame_height
        q2 = (5*self.frame_height *p[0])/self.frame_width - 3*p[1] -self.frame_height
        if((q1>0 and q2>0) or (q1<0 and q2<0)):
            return False
        return True
    def point_in_box(self,p,box):
        return p[0]>box.xmin and p[0]<box.xmax and p[1]>box.ymin and p[1]<box.ymax
       