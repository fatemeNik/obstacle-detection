import cv2
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
# from temp import SpaceCapture
from playsound import playsound
from threading import Thread
import torch


            # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
VOICE_PATH = r'C:/Users/PicoNet/source_code_tracking/voices/'
MODEL_PATH = r'C:/Users/PicoNet/source_code_tracking/yolov5'
# model = torch.hub.load(MODEL_PATH, 'yolov5s',source='local', force_reload=True)
model = torch.hub.load(MODEL_PATH, 'yolov5l',source='local')


voices ={'person':'person.mp3','bicycle':'bicycle.mp3','toothbrush':'toothbrush.mp3','hair drier':'hair drier.mp3','teddy bear':'teddy bear.mp3','scissors':'scissors.mp3',     
    'vase':'vase.mp3','clock':'clock.mp3','book':'book.mp3','refrigerator':'refrigerator.mp3','sink':'sink.mp3',
    'toaster':'toaster.mp3','oven':'oven.mp3','microwave':'microwave.mp3','cell phone':'cell phone.mp3','keyboard':'keyboard.mp3',
    'remote':'remote.mp3','mouse':'mouse.mp3','laptop':'laptop.mp3','tv':'tv.mp3','toilet':'toilet.mp3',
    'diningtable':'diningtable.mp3','bed':'bed.mp3','potted plant':'pottedplant.mp3','couch':'sofa.mp3','chair':'chair.mp3',
    'cake':'cake.mp3','donut':'donut.mp3','pizza':'pizza.mp3','hot dog':'.mp3','hot dog':'.mp3',
    'carrot':'carrot.mp3','broccoli':'broccoli.mp3','orange':'orange.mp3','sandwich':'sandwich.mp3','apple':'apple.mp3',
    'banana':'banana.mp3','bowl':'bowl.mp3','spoon':'spoon.mp3','knife':'knife.mp3','fork':'fork.mp3',
    'cup':'cup.mp3','wine glass':'wine glass.mp3','bottle':'bottle.mp3','tennis racket':'tennis racket.mp3','surfboard':'surfboard.mp3',
    'skateboard':'skateboard.mp3','baseball glove':'baseball glove.mp3','baseball bat':'baseball bat.mp3','kite':'kite.mp3','sports ball':'sports ball.mp3',
    'snowboard':'snowboard.mp3','skis':'skis.mp3','frisbee':'frisbee.mp3','suitcase':'suitcase.mp3','tie':'tie.mp3',
    'handbag':'handbag.mp3','umbrella':'umbrella.mp3','backpack':'backpack.mp3','giraffe':'giraffe.mp3','zebra':'zebra.mp3',
    'bear':'bear.mp3','elephant':'elephant.mp3','cow':'cow.mp3','sheep':'sheep.mp3','horse':'horse.mp3',
    'dog':'dog.mp3','cat':'cat.mp3','bird':'bird.mp3','bench':'bench.mp3','parking meter':'parking meter.mp3',
    'stop sign':'stop sign.mp3','fire hydrant':'fire hydrant.mp3','traffic light':'traffic light.mp3','boat':'boat.mp3','truck':'truck.mp3',
    'train':'train.mp3','bus':'bus.mp3','airplane':'aeroplane.mp3','motorcycle':'motorbike.mp3','car':'car.mp3'  
}

def sound(file):
    playsound(file)

def detection(model,frame):
   
    results = model(frame)
    result = results.pandas().xyxy[0]
    return result

def draw_detection_box(result,frame):

    for i in range(len(result.index)):
            
        name,xmin,ymin,xmax,ymax =  result.at[i,'name'], int(result.at[i,'xmin']),int(result.at[i,'ymin']),int(result.at[i,'xmax']),int(result.at[i,'ymax'])
    
        w = xmax-xmin
        h = ymax-ymin
        distancei = ((2 * 3.14 * 180) / (w + h* 360) * 1000 + 3)
        cv2.putText(frame,"{}".format(distancei),(xmin,ymin+50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        output = [name , distancei]
        return output
        
             

def location(x,w,center_x,frame_xmid):

     if center_x > frame_xmid and int(x+w) > frame_xmid:
        playsound("right.mp3")
        print("right")
     elif center_x < frame_xmid and int(x)< frame_xmid:
        print("left")
        playsound("left.mp3")
     elif center_x == frame_xmid :
        print("front")
        playsound("front.mp3")



def distance_from_cam(distance):
    if distance > 50:
        print("safe")
        playsound("safe.mp3")
    elif 20<distance<50:
        print("near")
        playsound("near.mp3")
    elif distance<20:
        print("warning")
        playsound("warning.mp3")


def main(image):

    results = model(image)
    result = results.pandas().xyxy[0]
    for i in range(len(result.index)):
            
        name,xmin,ymin,xmax,ymax =  result.at[i,'name'], int(result.at[i,'xmin']),int(result.at[i,'ymin']),int(result.at[i,'xmax']),int(result.at[i,'ymax'])
    
        w = xmax-xmin
        h = ymax-ymin
        cx = int(xmin+xmax)/2
        distancei = ((2 * 3.14 * 180) / (w + h* 360) * 1000 + 3)
        cv2.putText(image,"{}".format(distancei),(xmin,ymin+50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
        cv2.putText(image,"{}".format(name),(xmin,ymin+100),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
        distance_from_cam(distancei)
        location(xmin,w,cx,320)
        playsound(voices[name])
      
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
    return image

def beep():
    playsound("beep.wav")

def run(detected):
    while detected== True:
        playsound("beep.wav")
        if detected == False:
            break
        

# cap = cv2.VideoCapture("test.mp4")
cap = cv2.VideoCapture(0)
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# camera = SpaceCapture()
while True:

        
        
        ret, frame = cap.read()
        # region = frame[280:480,0:200]
        print(frame.shape)

        key = cv2.waitKey(1)
        if key & key == 97:
        
            image = main(frame)

            cv2.imshow("frame",image)
            
            cv2.waitKey(3000)
                

      
        # pt1=(80,640)
        # pt2=(600,640)
        # pt3=(340,150)
        # triangle_cnt = np.array( [pt1, pt2, pt3] )

        # cv2.drawContours(frame, [triangle_cnt], 0, (0,255,0), 0)
        # region = frame[200:500,0:300]
        region = frame[100:480,155:515]
  

       

       
        blur = cv2.bilateralFilter(region,20,100,100)
        median = cv2.medianBlur(blur,9)
        laplacian = cv2.Laplacian(median,cv2.CV_64F)
        mask = object_detector.apply(laplacian)

        ret, mask = cv2.threshold(mask, 254, 255, 0)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours: 
            area =cv2.contourArea(cnt)
            detected = False
            if area > 200 :
                    # print("unknown obstacle")
                     # t = Thread(target=sound)
                    # t.start()
                    x, y, w, h = cv2.boundingRect(cnt)
                    if x+w<=600 and x+w>=80 and x>=80 and y+h<=480 and y+h>=150:
                         t1 = Thread(target = beep)
                         t1.daemon = True
                         t1.start()
                    
                         cx = int((x + x + w) / 2)
                         cy = int((y + y + h) / 2)
                    
                         cv2.circle(region,(cx,cy),5,(0,0,255),-1)
                         cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # cv2.imshow("roi",region)
        cv2.imshow("frame",frame)
        # cv2.imshow("mask",mask)
        if key & 0xFF == ord('q'):
            break




cap.release()
cv2.destroyAllWindows()
