import cv2 
import numpy as np

#Se accedde al video 
video = cv2.VideoCapture("highway_vid.mp4")


def procesamiento(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Corregir nombre de la variable
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_m = cv2.Canny(blur, 50, 150)
    return canny_m  # Devolver el resultado procesado, no el nombre de la función

def region_of_interest(frame):
    height= frame.shape[0]
    polygons= np.array([[ (200, height), (3100, height), (1400, 1750)]]) 
    mask=np.zeros_like(frame)
    cv2.fillPoly(mask,polygons,255)
    mask_image= cv2.bitwise_and(frame,mask)


    #return mask #si queremos regresar solo la mascara del poligono
    return mask_image  #regresamos solo el poligono donde mostrara el area de interes


while (video.isOpened()):
    #ret es un valor Boolenao que indica si el frame se leyo correctamente 
    #frame es el frame del archivo

    ret, frame = video.read()

    if not ret:
        video = cv2.VideoCapture("highway_vid.mp4")
        continue
    
    imagen_filtrada=procesamiento(frame)

    mask=region_of_interest(imagen_filtrada)
   
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


