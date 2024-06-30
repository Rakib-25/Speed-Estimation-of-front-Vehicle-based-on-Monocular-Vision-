import torch
import numpy as np
import cv2
import time
# import pafy

import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tracker import Tracker

Conf_threshold = 0.4
NMS_threshold = 0.4

focal_lenth = []


focal_length1 = {'Motor-bike':0.23, 'Car':0.212, 'Truck':0.23, 'Easy-bike': 0.2, 'Van':0.2, 'By-cycle':0.21, 'Rickshaw':0.2,'Bus':.23,'CNG':0.2,} 
count = 0
speed = {}
speed [0] = 0
speed[1] = 0
speed[2] = 0
speed[3] = 0
speed[4] = 0

fps = 0
Bheight = []

height = {
  "CNG": 70,
  "Easy-bike": 72,
  "Bus": 9*12+8,
  "Truck": 9*12+8,
  "Motor-bike": 66,
  "By-cycle": 45,
  "Car": 5*12,
  "Van": 50,
  "Rickshaw": 6*12+5,
}


ownSpeed = 0



# model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)

custom_model_path = 'for_training_yolo/best.pt'

# Loaded my custom trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_model_path, force_reload=True)


tracker = Tracker()



def focal_lenght_finder(distance, real_object_height, height_in_frmae):
    height_in_frame = height_in_frmae *0.0002645833
    focal_length = (distance* height_in_frame) / (real_object_height+height_in_frame)
    return focal_length



def distance_finder(focal_lenth, real_object_height, height_in_frmae):
    height_in_frmae = height_in_frmae *0.0002645833
    distance = (real_object_height/height_in_frmae+1)*focal_lenth
    return distance

def SpeedFinder(u,v,distance_u,distance_v,fps):
    #speed = s/t {s = distance, t = time}
    # s = distance_v - distance_u
    # time diff of two consiqutive frames = 1/fps
    
    speed = (distance_v-distance_u)*fps/(v-u) 
    speed = speed*3.6
    return speed
 
    

# #for Truck
# image_path = 'images/car_5.jpg'


# frame = cv2.imread(image_path)



video_path = 'Test_video/bus_20.MOV'
cap = cv2.VideoCapture(video_path)

# video_path = 'path_to_your_video.mp4'  # Path to your video file
# cap = cv2.VideoCapture(video_path)
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_video_path = 'Test_video/Bus_20kmh_tested.mp4' # Path to save the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1137, 640))





if not cap.isOpened():   
    print("Error: Could not open video.")
    exit()

speed_calc = {}




# Open the video file
video_capture = cv2.VideoCapture(video_path)

# # Initialize variables
# frame_count = 0
# start_time = 0

# # Read frames from the video and count them
# while True:
#     # Read a frame from the video
#     ret, frame = video_capture.read()

#     # Check if frame is successfully read
#     if not ret:
#         break

#     # Increment frame count
#     frame_count += 1

# # Calculate elapsed time
# end_time = 8
# total_time = (end_time - start_time)   # Total time in seconds

# # Calculate frames per second (FPS)
# fps = frame_count / total_time

# print(f"Frames processed: {frame_count}")
# print(f"Total time taken: {total_time:.2f} seconds")
# print(f"Frames per second (FPS): {fps:.2f}")

# # Release the video capture object
# video_capture.release()










for i in range(3, 30):
    # Construct the image filename
    filename =f'Bike_photosWithDistance/{i}m.jpg'
    if i%3 != 0:
        continue
    # Read the image using OpenCV
    image = cv2.imread(filename)
    
    desired_height = 640  # Set your desired size here

    # Calculate the aspect ratio of the image
    height1, width, _ = image.shape
    
    desired_width = (desired_height*width)/height1
    desired_width = int(desired_width)
    
    # Resize the image
    image = cv2.resize(image, (desired_width, desired_height))
    
        # Define the desired width or height (in pixels)
     
    results = model(image)
    
    # Process and draw bounding boxes on the image based on the detection results
    if results.pred[0] is not None:
        pred = results.pred[0]
        for det in pred:
            label = model.names[int(det[5])]
            
            # print(height[label])
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            heights = (y2-y1)
            # print(heights)
            
            focals = focal_lenght_finder(i,height[label]*0.0254,heights)
            focal_lenth.append(focals)
            Bheight.append(heights)
            break
    # cv2.imshow("Object Detection", image )
    # cv2.waitKey(0)  # Wait for a key press to close the window
       
#generate curve of focal length for object
# Example x and y values of the curve
x_values = Bheight # Example x values
y_values = focal_lenth  # Corresponding y values
# print(x_values)
# print(y_values)

# Perform polynomial interpolation
degree = 1  # Degree of the polynomial (quadratic curve)
coefficients = np.polyfit(x_values, y_values, degree)
curve_function = np.poly1d(coefficients)

# Test the curve function with some x valu



# # Check if the image was successfully read
#     if image is not None:
#         # Process the image (e.g., display, save, etc.)
#         cv2.imshow(f'Image {i}', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print(f'Error: Unable to read {filename}')


d =0
speed1 = 0
# commenting now

u = 0
j = 0
while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv2.imread('images/car_5.jpg')
    # j+=1
    if not ret:
        break

    count += 1
    if count<60 :
        continue 
    # if count >120 and count < 150:
    #     continue 
    # if count >180:
    #     continue
    
    desired_height = 640  # Set your desired size here

    # Calculate the aspect ratio of the image
    height1, width, _ = frame.shape
    
    desired_width = (desired_height*width)/height1
    desired_width = int(desired_width)
    
    # Resize the image
    frame = cv2.resize(frame, (desired_width, desired_height))
    # frame = cv2.resize(frame, (1280, 720))
    
    results = model(frame)

    detections = {}

    # Process and draw bounding boxes on the image based on the detection results
    if results.pred[0] is not None:
        pred = results.pred[0]
        for det in pred:
            
            label = model.names[int(det[5])]
            # print(label)
            conf = det[4].item()
            if conf <= 0.5:
                continue
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            detections[label] = (x1,y1,x2,y2)
            # box_width = x2 - x1   
            # box_height = y2 - y1
             
            # RealHeight = height[label] * 0.0254
            # distance = distance_finder(focal_length,RealHeight,box_height)
            # print(distance)
            #if label == "Car":
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #cv2.putText(frame, f"{label} ({distance:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, f"{label} ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
           
        #print(object_id)
        # name = []
        # i = 0
        # for label in detections.keys():
        #     name.append(label) 
        #     i+=1;
            
            
        
        i = 0
        object_id = tracker.update(detections,u)
        
        
        for x,y,w,h,id ,label in object_id:
            
            # label = name[i]
            # print(label)
            #   label = "Motor-bike"
            box_width = w
            box_height = h
            
            # print(box_height)
            RealHeight = height[label] * 0.0254
            # print(RealHeight)
            # focal_lenth = focal_lenght_finder(10,RealHeight,box_height)
            # print (focal_lenth)
            # focal_length = curve_function(box_height)
            distance = distance_finder(focal_length1[label],RealHeight,box_height)
            # print(distance)
            
            # if distance> 30:
            #     continue
            # print(distance)
            #id_num, name, frame_no = distance from camera
            
            speed_calc[id,label,u] = (distance)
            
            i+=1
            realSpeed = speed[id] + ownSpeed
            j+=1
            if j<30:
                realSpeed = 0
            try:
                mark = ''
                if realSpeed < 0:
                    mark = 'InComing'
                elif realSpeed > 0:
                    mark = 'OutGoing'
                else:
                    mark = 'Calculating or Stopped'
            except:
                realSpeed = 0
                
            # realSpeed = speed[id] + ownSpeed
            
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, f"({realSpeed:.2f}){label}({mark})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if u % 30 != 0:
                continue 
            print(speed_calc)
            try:
                d+=1
                speed[id] = SpeedFinder(u-30,u,speed_calc[id,label,u-30],speed_calc[id,label,u],30 )
                speed1 = speed [id]
                
                # if  SpeedFinder(u-5,u,speed_calc[id,label,u-5],speed_calc[id,label,u],30 )<0:
                #     d+=1
                #     speed1 =  speed1 + SpeedFinder(u-5,u,speed_calc[id,label,u-5],speed_calc[id,label,u],30 )    
                # else:
                
                #     speed1 =  speed1 + SpeedFinder(u-5,u,speed_calc[id,label,u-5],speed_calc[id,label,u],30 )    
                       
            except:
                speed[id] = speed1
            #     speed1 = speed1
            # print(speed1)
            
            # if u%30 != 0 or u < 1 :
            #     continue
            # speed[id] = speed1/d
            # print(d)
            
            
            

            # for obj_id, bbox in tracked_objects.items():
            #     #x1, y1, x2, y2 = map(int, bbox)
            #     cv2.putText(frame,f"{label} {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    u+=1
    # print(speed_calc)
    
    out.write(frame)
    # cv2.imwrite('extract/img.png',frame)
    # print(frame.shape)
    # Display the image with bounding boxes
    cv2.imshow("Object Detection", frame)
    # cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(0)  # Wait for a key press to close the window
cap.release()
out.release()
cv2.destroyAllWindows()


