import cv2 
import numpy as np

#Se accedde al video 
video = cv2.VideoCapture("highway_vid.mp4")


while (video.isOpened()):
    #ret es un valor Boolenao que indica si el frame se leyo correctamente 
    #frame es el frame del archivo

    ret, frame = video.read()

    if not ret:
        video = cv2.VideoCapture("highway_vid.mp4")
        continue
    
    hsvWhite = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit = np.array([0, 0, 200])
    upperLimit = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsvWhite, lowerLimit,upperLimit)
    edges = cv2.Canny(mask, 75, 150)

    lines= cv2.HoughLinesP(edges,1, np.pi/180, 30, maxLineGap=80)

    if lines is not None:
        for line in lines:
           x1,y1,x2,y2 = line[0] #Tomamos cero porque vamos a tomar el array dentro del array 
           cv2.line(frame, (x1,y1), (x2, y2), (0,0,255), 3)
    

    cv2.imshow("Test video", frame)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Bordes", edges)


    #se cierra presionando la ventana de ENTER 
    if cv2.waitKey(1) ==13:
        break

#Se libera el archivo y se cierra la ventana 
video.release()
cv2.destroyAllWindows()


