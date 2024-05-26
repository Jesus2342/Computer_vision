import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Se accede al video 
video = cv2.VideoCapture("highway_vid.mp4")

def procesamiento(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Corregir nombre de la variable
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_m = cv2.Canny(blur, 50, 150)
    return canny_m  # Devolver el resultado procesado, no el nombre de la función

def region_of_interest(frame):
    height= frame.shape[0]
    polygons= np.array([[ (500, height), (3100, height), (1500, 1750)]]) 
    mask=np.zeros_like(frame)
    cv2.fillPoly(mask,polygons,255)
    mask_image= cv2.bitwise_and(frame,mask)
    #return mask #si queremos regresar solo la mascara del poligono
    return mask_image  #regresamos solo el poligono donde mostrara el area de interes
    
     #dimesiones del poligono. 
    #Esquina inferior 2000
    #Esquina derecha  3500
    #punto vertice superior- 
    #El primer valor representa el punto medio del tringulo y el segundo la altura. 
    #Entre menor sea el valor de la altura, mayor sera el triangulo
def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2=line.reshape(4)
            cv2.line(line_image,(x1,y1), (x2,y2), (0,0,255), 10)
    return line_image




while (video.isOpened()):
    # ret es un valor Booleano que indica si el frame se leyó correctamente 
    # frame es el frame del archivo

    ret, frame = video.read()
    lane_image= np.copy(frame)
    
    if not ret:  # Verificar si se pudo leer el frame correctamente
        break

    frame_procesado = procesamiento(frame)  # Llamar la función con el frame actual
    cropped_image= region_of_interest(frame_procesado)

    #lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    lines= cv2.HoughLinesP(cropped_image,1, np.pi/180, 30, maxLineGap=100)
    #Numero de intersecciones en el espacio Hough para detectar una linea

    line_image =display_lines(lane_image, lines) #Detecccion de linea por Hough
    combo_image = cv2.addWeighted(lane_image,0.8, line_image,1,1)
    cv2.imshow("Video_met2",combo_image)


    # Mostrar el frame procesado utilizando matplotlib
    #plt.imshow(frame_procesado, cmap='gray')
    #plt.show()
  
    # Se cierra presionando la ventana de ENTER 
    if cv2.waitKey(1) == 13:
        break

# Se libera el archivo y se cierra la ventana 
video.release()
cv2.destroyAllWindows()
