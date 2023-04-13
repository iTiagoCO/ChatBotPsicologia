# Proyecto final - Chatbot de terapia (Psicologo)

# Santiago Poveda Garcia
# Luis Felipe Toro
# Python - TensorFlow 

import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
from tensorflow.python.framework import ops 
import json
import random
import pickle

#nltk.download('punkt')

#Lectura del archivo
with open("contenido.json") as archivo:
	datos = json.load(archivo)

palabras = []
tags = []
auxX=[]
auxY=[]


#Iteracion de contenido
for contenido in datos["intents"]:
		for patrones in contenido["patterns"]:
			auxPalabra = nltk.word_tokenize(patrones)
			palabras.extend(auxPalabra)
			auxX.append(auxPalabra)
			auxY.append(contenido["tag"])

			if contenido["tag"] not in tags:
				tags.append(contenido["tag"])


# Se pasan cada una de las palabras para que sean mas entendibles adicional != de "?"
palabras = [stemmer.stem(w.lower()) for w in palabras if w!="?"]
#Ordenamos la lista de palabras
palabras = sorted(list(set(palabras)))
#Igual con los tags
tags = sorted(tags)


#Declaramos la lista de entrenamiento como tal 
entrenamiento = []
salida = []
salidaVacia = [0 for _ in range(len(tags))]

#Recorremos lsta auxiliar de palabras con enumerate es decir + los indices
for x, documento in enumerate(auxX):
	cubeta=[]
	auxPalabra= [stemmer.stem(w.lower())for w in documento] 
	#1 si esta, 0 si no esta
	for w in palabras: 
		if w in auxPalabra:
			cubeta.append(1)
		else:
			cubeta.append(0)
	filaSalida = salidaVacia[:]
	#En la posicion del indice del tag vamos a obtener el contenido de Y en cada uno de sus indices
	# y luego le vamos a asignar un 1 : sera una lista aparte.
	filaSalida[tags.index(auxY[x])] = 1
	entrenamiento.append(cubeta)
	salida.append(filaSalida)

# Las listas las convertiremos en arreglos de numpy
entrenamiento = numpy.array(entrenamiento)
salida = numpy.array(salida)


#Comenzar a entrenar 
ops.reset_default_graph()

#Entrada - Entrenamientos
red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
#10 neuronas - 2 Columnas | Entrada - Salida
red= tflearn.fully_connected(red,10) #Completamente conectado - Hidden Layers van a estar dando predicciones
red = tflearn.fully_connected(red,10)
#Neuronas x Salida
red = tflearn.fully_connected(red,len(salida[0]),activation="softmax") #Saludo | despedida | 2 neuronas

#Probabilidades de que tan eficiente es nustro tag
red = tflearn.regression(red)


#Creamos el modelo

modelo = tflearn.DNN(red)
modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=10,show_metric=True) 

														#n_epoch Cantidad de veces que vera nuestro modelo el dato| 
														#batch_size cantidad de entradas (Varia)
														#Varia tambien el numero de neuronas x las entradas 
modelo.save("modelo.tflearn")


def mainBot():
	while True:
		entrada = input("Tu: ")
		cubeta = [0 for _ in range(len(palabras))]
		entradaProcesada = nltk.word_tokenize(entrada)
		entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]

		for palabraIndividual in entradaProcesada:
			for i, palabra in enumerate(palabras):
				if palabra == palabraIndividual:
					cubeta[i] = 1
		resultados = modelo.predict([numpy.array(cubeta)])
		
		resultadosIndices = numpy.argmax(resultados)
		tag = tags[resultadosIndices]

		for tagAux in datos ["intents"]:
			if tagAux["tag"] == tag:
				respuesta = tagAux["responses"]

		print("Chatbot: ",random.choice(respuesta))
mainBot()