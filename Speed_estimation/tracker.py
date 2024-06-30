import math
from collections import Counter
from kalman_filter import KalmanFilter
c_points = {}
bounding_box = {}
common_label = {}
labels = []

kf = KalmanFilter()
 
 
class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    
    
    def calculate_overlap_area(self,bbox1, bbox2):
        # Extract coordinates of the bounding boxes
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Determine coordinates of the intersection rectangle
        intersection_x1 = max(x1, x2)
        intersection_y1 = max(y1, y2)
        intersection_x2 = min(x1 + w1, x2 + w2)
        intersection_y2 = min(y1 + h1, y2 + h2)
        
        # Calculate width and height of the intersection rectangle
        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        
        # Calculate area of the intersection rectangle
        intersection_area = intersection_width * intersection_height
        
        # Calculate coordinates of the union rectangle
        union_x1 = min(x1, x2)
        union_y1 = min(y1, y2)
        union_x2 = max(x1 + w1, x2 + w2)
        union_y2 = max(y1 + h1, y2 + h2)
        
        # Calculate width and height of the union rectangle
        union_width = union_x2 - union_x1
        union_height = union_y2 - union_y1
        
        # Calculate area of the union rectangle
        union_area = union_width * union_height
        
        # Calculate overlapping area ratio
        if union_area == 0:
            overlap_ratio = 0
        else:
            overlap_ratio = intersection_area / union_area
    
        return overlap_ratio
    
    
    
    
    
    
    
    
    
    def update(self, objects_rect,i):
        # Objects boxes and ids
        objects_bbs_ids = []
        
        save_id = {}

        # Get center point of new object
        if not objects_rect:
            for id in c_points:
                cx,cy,predictMin,predictMax,label = c_points[id]
                objects_rect[label] = predictMin[0],predictMin[1],predictMax[0],predictMax[1]
            print("here",objects_rect)
        
        for label in objects_rect:
            
            x, y, x2, y2 = objects_rect[label]
            w = x2-x
            h = y2 - y
            # cx = (x + x + w) // 2
            # cy = (y + y + h) // 2
            cx = (x + x2) // 2
            cy = (y + y2 ) // 2
            bbox = x,y,w,h
            predictMin = kf.predict(x,y)
            predictMax = kf.predict(x2,y2)
            
            
            
            
            # Find out if that object was detected already
            same_object_detected = False
#             for id, pt in self.center_points.items():
#                 print(id,pt)
#                 dist = math.hypot(cx - pt[0], cy - pt[1])

#                 if dist < 35:
#                     self.center_points[id] = (cx, cy)
#                     c_points[id] = (cx,cy)
# #                    print(self.center_points)
#                     objects_bbs_ids.append([x, y, w, h, id])
#                     same_object_detected = True
#                     break
#             print("sesh")
            for id, pt in c_points.items():
                #checking most freequent label
                if i == 10:
                    
                    for id , j in common_label:
                        # print(j)
                        labels.append(common_label[id,j])
                    # Count the occurrences of each label
                    label_counts = Counter(labels)

                    # Find the most common label
                    label = label_counts.most_common(1)[0][0]
                x1,y1 = pt[2]
                x3,y3 = pt[3]
                # if i< 10:
                dist = math.hypot(cx - pt[0], cy - pt[1])
                # else:
                # dist = math.hypot(cx - (x1+x3)/2, cy - (y1+y3)/2)
                # print(dist)
                # bbox = x1,y1,x3-x1,y3-y1
                ratio = self.calculate_overlap_area(bounding_box[id],bbox)
                # print(bbox)
                # print(bounding_box[id])
                if cx >700 :
                    save_id[id] = label
                elif dist < 35:
                    if ratio>.5:
                        # print(ratio)
                        if  (i > 10) and (label != common_label[id,i-1]):
                            label = common_label[id,i-1]
                        c_points[id] = (cx,cy,predictMin,predictMax,label)
                        bounding_box[id] = bbox
    #                    print(self.center_points)
                        common_label[id,i] = label
    
                        objects_bbs_ids.append([x, y, w, h, id,label])
                        same_object_detected = True
                        break
                    
                
                        
            # for id in save_id:
            #     try:
                    
            #         del c_points[id]
            #     except:
            #         continue
            # save_id.clear()
            # New object is detected we assign the ID to that object
            if same_object_detected is False and cx<900:
                # self.center_points[self.id_count] = (cx, cy)
                c_points[self.id_count] = (cx,cy,predictMin,predictMax,label)
                bounding_box[self.id_count] = bbox
                common_label[self.id_count,i] = label
                
                objects_bbs_ids.append([x, y, w, h, self.id_count,label])
                self.id_count += 1
           
        # print(common_label)
        # Clean the dictionary by center points to remove IDS not used anymore
        # new_center_points = {}
        # for obj_bb_id in objects_bbs_ids:
        #     _, _, _, _, object_id = obj_bb_id
        #     #center = self.center_points[object_id]
        #     center = c_points[object_id]
            
        #     new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        # self.center_points = new_center_points.copy()
        # print(common_label)
        for id in c_points:
            try:
                asd = common_label[id,i]
            except:
                common_label[id,i] = common_label[id,i-1]
                
                
                
        # print(common_label) 
        
        print(c_points)      
            
        return objects_bbs_ids