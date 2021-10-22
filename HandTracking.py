import cv2
import mediapipe as mp
import time


#to start videocam
cap = cv2.VideoCapture(0)

#hand detection model
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#drawing hand points(dots) provided by mediapipe
mpDraw = mp.solutions.drawing_utils

#frame rate
pTime = 0
cTime = 0

while True:
    sucess, img = cap.read()

    #creating RBG Image for hands class
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            #getting information from hand class
            #each id will have a corresponding landmark (x,y,z)
            #need x and y coorniates to find location of hand
            for id, lm in enumerate(handLms.landmark):
                print(id,lm)
                #converting decial values of x, y to pixel
                #getting width and heigh
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id== 4:
                    #15 refers to the radius of the tip
                    cv2.circle(img,(cx,cy), 15, (255,0,255), cv2.FILLED)


            #mpHands.HAND_CONNECTIONS is used to draw the lines to
            #connect the dots made in hand
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)


    cTime=time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime

    #to display fps on screen
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)



    cv2.imshow("Image",img)

    cv2.waitKey(1)
