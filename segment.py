import cv2
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
import os
import kornia
from PIL import Image
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

LIVE_FEED = True

class Emoji_Classifier(nn.Module):
      def __init__(self):
          super(Emoji_Classifier, self).__init__()
          self.name = "net"
          self.conv1 = nn.Conv2d(3, 10, 5, 2) #in_channels, out_chanels, kernel_size
          self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
          self.conv2 = nn.Conv2d(10, 26, 5, 2) #in_channels, out_chanels, kernel_size
          self.fc1 = nn.Linear(26 * 13 * 13, 120)
          self.fc2 = nn.Linear(120, 8)

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = x.view(-1, 26 * 13 * 13)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          x = x.squeeze(1) #Flatten to batch size
          return x

#Open Camera object
cap = cv2.VideoCapture(0)

# load emojis 
tu = cv2.imread('./emojis/tu.png')
fi = cv2.imread('./emojis/fi.png')
ro = cv2.imread('./emojis/ro.png')
ok = cv2.imread('./emojis/ok.png')
pt = cv2.imread('./emojis/pt.png')
up = cv2.imread('./emojis/up.png')
no = cv2.imread('./emojis/no.png')


def addOverlayEmoji(img1, img2, topBoundary, rightBoundary, scale_percent, offset):
    scale_percent = round(scale_percent, 2)
    # print(scale_percent)
    width = max(int(img2.shape[1] * scale_percent), 1)
    height = max(int(img2.shape[0] * scale_percent), 1)
    dim = (width, height)
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[topBoundary-rows-offset:topBoundary-offset, rightBoundary-cols:rightBoundary]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[topBoundary-rows-offset: topBoundary-offset, rightBoundary-cols:rightBoundary] = dst
    return img1


def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
 

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)


model = torch.load("./model_dict", map_location=torch.device('cpu'))
# model.load_state_dict(torch.load("./model_dict", map_location=torch.device('cpu')))
model.eval()

def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

def show_image(image):
    # Convert image to numpy
    image = image.numpy()
    
    # Un-normalize the image
    # image[0] = image[0] * 0.226 + 0.445
    
    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))

def test_images(model, PATH):
    for picture in os.listdir(PATH):
        path_to_img = PATH + "/" + picture
        if ".jpg" not in path_to_img and ".png" not in path_to_img:
            continue
        img = cv2.imread(path_to_img)
        loader = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = Image.open(path_to_img)
        image_tensor = loader(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model(input)
        plot(output, picture)
        pred = output.data.cpu().numpy().argmax()
   

def process(image):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transformations(image).float().cpu()
    img_t = img_t.unsqueeze(0)
    # image = image.transpose((2, 0, 1))
    # image = image/255
    
    
    # print(cropped_image.shape)
    # cropped_image = cropped_image.transpose((2, 0, 1))
    return img_t

def plot(output, picture):
    sm = torch.nn.Softmax()
    probabilities = sm(output) 
    objects = ('fi', 'no', 'ok', 'pt', 'ro', 'tu', 'up')
    y_pos = np.arange(len(objects))
    performance = []
    for i in range(len(objects)):
        performance.append(probabilities[0][i].item())
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xlabel('Gesture')
    plt.ylabel('Probability')
    plt.title(picture)
    plt.show()

classes_map = ['fi', 'no', 'ok', 'pt', 'ro', 'tu', 'up']

classes = {
    "fi": fi,
    "no": no,
    "ok": ok,
    "pt": pt,
    "ro": ro,
    "tu": tu,
    "up": up
}

if not LIVE_FEED:
    test_images(model, "./test")

else:
    while(1):
        #Measure execution time 
        start_time = time.time()
        
        #Capture frames from the camera
        ret, frame = cap.read()
        height, width, channels = frame.shape

        rightFrame = frame[:, 0: int(width/2)]
    
        #Blur the image
        blur = cv2.blur(rightFrame,(3,3))
        
        #Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Perform morphological transformations to filter out the background noise
        #Dilation increase skin color area
        #Erosion increase skin color area

        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)
        
        #Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        
        #Find Max contour area (Assume that hand is in the frame)
        max_area=100
        ci=0	
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i  
                
        #Largest area contour 			  
        cnts = contours[ci]

        #Find moments of the largest contour
        moments = cv2.moments(cnts)
        
        #Central mass of first order moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)    

        #Draw center mass
        cv2.circle(frame,centerMass,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)  
        leftBounary = max(int(cx - 112*2.3), 0)
        rightBoundary = min(int(cx + 112*2.3), width)
        topBoundary = min(int(cy + 112*2.3), height)
        bottomBoundary = max(int(cy - 112*2.3), 0)
        cropped_image = frame[bottomBoundary:topBoundary, leftBounary:rightBoundary]
        cropped_image = cv2.resize(cropped_image, (224,224))

        frame = cv2.rectangle(frame, (leftBounary,bottomBoundary), (rightBoundary, topBoundary), (0, 0, 0), 3)
        
        img_t=process(cropped_image)
        output = model(img_t)
        sm = torch.nn.Softmax()
        probabilities = sm(output) 
        # print(probabilities[0][0].item())
        pred = output.max(1, keepdim=True)[1]
        for index, name in enumerate(classes_map):
            frame = addOverlayEmoji(frame, classes[name], topBoundary, rightBoundary, probabilities[0][index].item(), 70*index)

        # cv2.imshow("Hand2", torch_img.numpy())
        cv2.imshow('Emoji', frame)
        cv2.waitKey(1)


    cap.release()
    cv2.destroyAllWindows()