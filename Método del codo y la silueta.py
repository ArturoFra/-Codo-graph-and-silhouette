#!/usr/bin/env python
# coding: utf-8

# # Método del codo y el factor de la silueta del claustering

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics 
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score


# In[4]:


x1=np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2=np.array([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3])
X = np.array(list(zip(x1,x2))).reshape(len(x1), 2)


# In[5]:


plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title("Dataset a clasificar")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x1,x2)
plt.show()


# In[12]:


max_k=10
K=range(1,max_k)
ssw=[]
color_palette = [plt.cm.Spectral(float(i)/max_k)for i in K]
centroid = [sum(X)/len(X) for i in K]
sst = sum(np.min(cdist(X,centroid, "euclidean"), axis=1))

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    centers=pd.DataFrame(kmeanModel.cluster_centers_)
    labels = kmeanModel.labels_
    
    
    
    ssw_k=sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
    ssw.append(ssw_k)
    
    label_color= [color_palette[i] for i in labels]
    
    
    ##Fabricar silueta para cada cluster pero donde k=1 o k=len(X) no hay silueta 
    if 1<k<len(X):
        #Crear un subplot de una fila y dos columnas 
        fig, (axis1,axis2) = plt.subplots(1,2)
        fig.set_size_inches(20,8)
        
        #El primer subplot tendrá valores entre -1 y 1
        axis1.set_xlim([-0.1, 1.0])
        #El numero de clusters determinara el tamaño de barra 
        #El coeficiente n_cluster+1 * 10 sera el espacio en blanco entre siluetas 
        
        axis1.set_ylim([0, len(X) + (k+1) * 10])
        silhouette_avg=silhouette_score(X, labels)
        print("Para k = ",k," el promedio de la silueta es de:", silhouette_avg)
        sample_silhouette_values = silhouette_samples(X,labels)
        y_lower=10
        for i in range(k):
            #Agregamos la silueta del cluster i-ésimo
            ith_cluster_sv= sample_silhouette_values[labels== i]
            print(" -Para i=", i+1, "La silueta del cluster vale: ", np.mean(ith_cluster_sv))
            #ordenar las siluetas de forma descendente 
            ith_cluster_sv.sort()
            
            ith_cluster_size = ith_cluster_sv.shape[0]
            y_upper = y_lower + ith_cluster_size
            
            
            color= color_palette[i]
            
            axis1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sv, facecolor=color, alpha = 0.7)
            
            
            #Etiquetamos dicho cluster en el centro 
            
            axis1.text(-0.05, y_lower + 0.5 + ith_cluster_size, str(i+1))
            
            
            #Calculamos el nuevo y_lower para el nuevo cluster 
            
            y_lower = y_upper+10 # Dejamos 10 posiciones sin muestra 
            
        axis1.set_title("Representación de la silueta para K=%s"%str(k))
        axis1.set_xlabel("S(i)")
        axis1.set_ylabel("ID del cluster")
            
            
        ##Fin de la representación del cluster
            
    #Plot de los Kmeans con lo puntos respectuvos 
    plt.plot()
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.title("Clustering para k = %s"%str(k))
    plt.scatter(x1,x2, c=label_color)
    plt.scatter(centers[0],centers[1], c=color_palette[k], marker = "x")
    plt.show()
    
    
    
    


# In[15]:


#Representación del codo

plt.plot(K, ssw, "bx-")
plt.xlabel("k")
plt.ylabel("Ssw(k)")
plt.title("Técnica del codo para encontrar el k óptimo")
plt.show()


# In[16]:


#Representación del codo normalizada 
plt.plot(K, 1-ssw/sst, "bx-")
plt.xlabel("k")
plt.ylabel("1-Ssw(k)")
plt.title("Técnica del codo normalizado para encontrar el k óptimo")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




