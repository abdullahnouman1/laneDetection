import cv2
import numpy as np
import matplotlib.pyplot as plt

#Process the image to help us select the lane we want to detect and draw lines on
def processImage(image):
    height = image.shape[0]
    width = image.shape[1]

    #Create vertices of the lane we want
    triangle_vert = [(0,height), (5*width/10, 6*height/10), (width,height)]
    
    #Gray scales the image to isolate important features
    gray_scale = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #Detects edges in the image
    canny = cv2.Canny(gray_scale,100,150)
    #Crops out our region of interest which is the right lane
    cropped = regionOfInterest(canny,np.array([triangle_vert],np.int32))
    
    return cropped

#Isolates the region of interest and masks everything else
def regionOfInterest(image, vertices):
    #Create a black image
    mask_img = np.zeros_like(image)
    #Draws our region of interest onto the black image with a color
    cv2.fillPoly(mask_img, vertices, 255)
    #Creates a trace of the lane and vertices
    masked_bits = cv2.bitwise_and(image, mask_img)
    
    return masked_bits

#Draw the lines on a blank image using the Hough Transform function and combine it with the original image
def drawLines(image, lines):
    image = np.copy(image)
    #Creates a blank image
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    #For each line, show every line on the blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2,y2), (255,0,0), thickness=10)

    #Combine the lines on the blank image to our original image
    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    
    return image

#Passes in the video we want to show the lane detection on
cap = cv2.VideoCapture('machineLearning/test2.mp4')

#Create the loop to display the lane detection video
while(cap.isOpened()):
    _,frame = cap.read()

    #Call our proccess function and pass in the frame
    cropped_image = processImage(frame)
    #Use the HoughLines algorithm to detect lines on the frame
    get_lines = cv2.HoughLinesP(cropped_image, rho=6, threshold=60, theta=np.pi/180, minLineLength=50, maxLineGap=150, lines=np.array([]))
    #The image that has the result of the code
    finalImg = drawLines(frame, get_lines)

    #Display the result
    cv2.imshow('Lane Detection', finalImg)
    if cv2.waitKey(1) & 0xFF == 27:
       break