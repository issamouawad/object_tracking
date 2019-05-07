from detection import Detection
import numpy as np
from scipy.spatial import distance
def get_distance(v1,v2):
    
    if(len(v1)==len(v2)):
        dist = distance.euclidean(np.array(v1),np.array(v2))
    else:
        dist = -1
    if(dist>1000):
        dist=30
    return dist
def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    

    top_left_x = min(point[0][0] for point in points)
    top_left_y = min(point[0][1] for point in points)
    bot_right_x = max(point[0][0] for point in points)
    bot_right_y = max(point[0][1] for point in points)

    center_x = (top_left_x+bot_right_x)/2
    center_y = (top_left_y+bot_right_y)/2
    return [center_x,center_y,bot_right_x-top_left_x,bot_right_y-top_left_y]
def load_detections(dataset,detector,boat_class,min_conf):
    text_file_path = "detections/%s/%s.txt"%(dataset,detector)
    f = open(text_file_path,"r")
    line = f.readline()
    detections={}
    comps = []
    while(line):

        line = line.replace("\n", "")
        comps = line.split(",")
        
        if(int(comps[2])==boat_class and float(comps[3])>min_conf):
            
            if(not comps[0] in detections):
                detections[comps[0]]=[]
            if (not (dataset=='graal_2' and int(comps[4])>270 and int(comps[4])<740 and int(comps[5])>540)):
                
                detections[comps[0]].append(Detection(comps[3],comps[4:8],comps[8:]))
            
        line=f.readline()
        
    f.close()
    
    return detections
def iou(a, b):
	epsilon=1e-5

	x1 = max(a[0], b[0])
	y1 = max(a[1], b[1])
	x2 = min(a[2], b[2])
	y2 = min(a[3], b[3])

	# AREA OF OVERLAP - Area where the boxes intersect
	width = (x2 - x1)
	height = (y2 - y1)
	# handle case where there is NO overlap
	if (width<0) or (height <0):
		return 0.0
	area_overlap = width * height

	# COMBINED AREA
	area_a = (a[2] - a[0]) * (a[3] - a[1])
	area_b = (b[2] - b[0]) * (b[3] - b[1])
	area_combined = area_a + area_b - area_overlap

	# RATIO OF AREA OF OVERLAP OVER COMBINED AREA
	iou = area_overlap / (area_combined+epsilon)
	return iou    



