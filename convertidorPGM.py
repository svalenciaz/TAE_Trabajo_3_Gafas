import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

#plt.imshow(np.reshape(data[0],data[1])) # Usage example
# https://stackoverflow.com/questions/46944048/how-to-read-pgm-p2-image-in-python

carpetas = ['an2i', 'at33', 'boland', 'bpm', 'ch4f', 'cheyer', 'choon',
            'danieln', 'glickman', 'karyadi', 'kawamura', 'kk49', 'megak',
            'mitchell', 'night', 'phoebe', 'saavik', 'steffi', 'sz24', 'tammo']

direcciones_archivos =[]
for nombre in carpetas:
    direcciones_archivos += glob.glob(".\\faces\\"+nombre+"\\*.pgm")

entera_imagenes = []
entera_nombres = []
entera_sin_gafas_imagenes = []
entera_con_gafas_imagenes = []
entera_sin_gafas_nombres = []
entera_con_gafas_nombres = []

media_imagenes = []
media_nombres = []
media_sin_gafas_imagenes = []
media_con_gafas_imagenes = []
media_sin_gafas_nombres = []
media_con_gafas_nombres = []

cuarto_imagenes = []
cuarto_nombres = []
cuarto_sin_gafas_imagenes = []
cuarto_con_gafas_imagenes = []
cuarto_sin_gafas_nombres = []
cuarto_con_gafas_nombres = []


for direccion in direcciones_archivos:
    imagen = cv2.imread(direccion)
    nuevo_nombre = direccion.split('\\')[3].replace(".pgm","").split("_")
    if "4" in nuevo_nombre:
        cuarto_imagenes.append(imagen)
        cuarto_nombres.append(nuevo_nombre[1:4])
        if nuevo_nombre[3] == 'open':
            cuarto_sin_gafas_imagenes.append(imagen)
            cuarto_sin_gafas_nombres.append(nuevo_nombre[1:3])
        else:
            cuarto_con_gafas_imagenes.append(imagen)
            cuarto_con_gafas_nombres.append(nuevo_nombre[1:3])
    elif "2" in nuevo_nombre:
        media_imagenes.append(imagen)
        media_nombres.append(nuevo_nombre[1:4])
        if nuevo_nombre[3] == 'open':
            media_sin_gafas_imagenes.append(imagen)
            media_sin_gafas_nombres.append(nuevo_nombre[1:3])
        else:
            media_con_gafas_imagenes.append(imagen)
            media_con_gafas_nombres.append(nuevo_nombre[1:3])
    else:
        entera_imagenes.append(imagen)
        entera_nombres.append(nuevo_nombre[1:])
        if nuevo_nombre[3] == 'open':
            entera_sin_gafas_imagenes.append(imagen)
            entera_sin_gafas_nombres.append(nuevo_nombre[1:3])
        else:
            entera_con_gafas_imagenes.append(imagen)
            entera_con_gafas_nombres.append(nuevo_nombre[1:3])
    
plt.imshow(entera_imagenes[7])
print(entera_nombres[7])
print(media_con_gafas_nombres[7])
print(len(entera_nombres))
print(len(entera_con_gafas_nombres))
print(len(entera_sin_gafas_nombres))
print(len(media_nombres))
print(len(media_con_gafas_nombres))
print(len(media_sin_gafas_nombres))
print(len(cuarto_nombres))
print(len(cuarto_con_gafas_nombres))
print(len(cuarto_sin_gafas_nombres))



#https://stackoverflow.com/questions/33369832/read-multiple-images-on-a-folder-in-opencv-python/33371454