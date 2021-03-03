# -*- coding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import fonctions as fc 
import fprediction as fp
import joblib
import os
import pandas as pd
import numpy as np


def dataset ():
     st.title("**Datasets**")
     st.write( "Le jeu de données utilisé dans ce projet est le 'development dataset'\
               publié dans le cadre du DCASE 2020 Challenge Task 2, consistant en des bruits\
               de fonctionnement normaux / anormaux de 6 types de machines réelles. On dispose de 3 ou 4 identifiants différents par machine.\n\n")
     page = st.selectbox("Veuillez choisir une machine pour écouter le son produit", ["-", "ToyCar", "ToyConveyor", "fan", "pump","slider","valve"])
     if page != "-" :
      st.audio("./Fichiers_son/"+page+"_normal_id_00_00000000.wav", format='audio/wav', start_time=0)
     if st.checkbox("Afficher la distribution des types de machine dans l'ensemble d’entraînement"):
      st.image("./Images/distribution_train.png", width=550)
     if st.checkbox("Afficher la distribution des types de machine dans l'ensemble de test"):
      st.image("./Images/distribution_test.png", width=550)
 
def graphique(file):
     st.set_option('deprecation.showPyplotGlobalUse', False)
     audio, fe =fc.load_audio(file)
     st.subheader('Série temporelle du signal audio')
     plt.figure(figsize=(8,4))
     fig=fc.plot_audio(file)
     st.pyplot(fig)
     st.subheader('Représentation fréquentielle du signal audio')
     plt.figure(figsize=(8,4))
     fig2=fc.plot_spectrogram(audio,fe)
     st.pyplot(fig2)
     st.subheader("Représentation fréquentielle avec l'échelle de Mel du signal audio")
     plt.figure(figsize=(8,4))
     fig3=fc.plot_logMelSpectrogram(audio,fe)
     st.pyplot(fig3)
     st.subheader("Représentation Mel Frequency Energy Coefficients (MFECs)")
     plt.figure(figsize=(8,4))
     fig4=fc.plot_logMelEnergies(audio, fe)
     st.pyplot(fig4)

       
def visualisation ():
     st.title("**Analyse exploratoire**") 
     page = st.selectbox( "Veuillez choisir une machine pour visualiser les différentes représentations du signal", ["-", "ToyCar", "ToyConveyor", "fan", "pump","slider","valve"])
     if page != "-" : 
      graphique("./Fichiers_son/"+page+"_normal_id_00_00000000.wav")
    

def modelisation ():
     st.title("**Modélisation**")
     st.header('La classification d’identifiants machines (IDs)')
     st.write("Le modèle est d’abord entraîné à classifier les différents identifiants machines.\
              Puis, pour un son provenant d’un couple (machine, ID) donné, si le modèle le classe \
              dans une catégorie différente i.e. une machine (ou un ID) différente, \
              alors ce son est considéré comme anormal.")
     st.subheader('Architecture')
     st.image('./Images/modele_classification.png', width=700)
     
     if st.checkbox('Afficher la description du modèle de classification'):
         st.write("- 2 couches Conv2D avec 32 et 64 filtres, \
                      une taille de kernel de (5,5), des strides (2,1) et un padding ‘same’.\
                      Chacune de ces couches est suivie d’une activation ReLU et d’un Batch Normalization."
                  "\n\n"  
                  " - 3 couches Conv2D avec 128, 256 et 512 filtres, \
                      une taille de kernel de (3,3), des strides (1,1) et un padding ‘same’. \
                      Après chaque couche, un ReLU, un Batch Normalization et un MaxPooling2D de taille 2."
                 "\n\n"
                 "  - 1 couche d'aplatissement"
                 "\n\n"
                 "  - 2 couches Dense avec 512 neurones, puis 2 autres couches Dense avec 1024 neurones.\
                      Chaque couche Dense est suivie d’un ReLU, un Batch Normalization et un Dropout à 0.2.\
                      Une dernière couche à 23 neurones (nombre d’IDs machines) et une activation softmax ")
     st.header('L’autoencoder convolutionnel')
     st.write("Le but est d’avoir le même nombre d’input et d’output. On encode les entrées\
              pour isoler les particularités des sons normaux et leurs relations, puis on décode \
              c’est-à-dire qu’on reconstruit les données de sortie avec les éléments les plus significatifs. \
              L’erreur de reconstruction est alors utilisée pour conclure si le son en entrée est normal ou anormal.")

     st.subheader('Architecture')
     st.image('./Images/modele_autoencoder.png', width=560)
     
     if st.checkbox("Afficher la description de l'autoencoder"):
         st.write("- encoder : 5 couches de Conv2D avec respectivement 32, 64, 128, 256 et  512 filtres,  \
                       une taille de kernel de (5,5) pour les trois premières et (3,3) les deux suivantes,\
                       et des foulées de (1, 2), (1, 2), (2, 2), (2, 2), et (2, 2).\
                       Après chaque couche, un Batch Normalization et une activation LeakyReLU avec un alpha à 0,4."
                  "\n\n"  
                  " - Le goulot d’étranglement consiste en une couche Conv2D de 40 filtres. "
                  "\n\n"
                  " - decoder est le transposé de l’encoder : Conv2DTransposed, Batch Normalization et une activation LeakyReLU avec un alpha à 0,4")
           
              
def prédiction ():
     st.set_option('deprecation.showPyplotGlobalUse', False)
          
     st.title("**Prédiction**")
     page = st.selectbox( "Sur quelle machine souhaitez-vous faire une prédiction ?", ["-", "ToyCar", "ToyConveyor", "fan", "pump","slider","valve"])
     
     if page != "-": 
      df=pd.read_csv('./machinesCSV/'+page+'.csv', sep=';')
      fichier = st.selectbox( "Et choisissez un fichier audio", df.index)
      df1 = df.filepath[fichier]
     
      # prediction of label:
      pred=fp.predicted_class(df1, page)  
      if pred == 'anomaly':
        st.write('Le modèle prédit que ce son est **anormal**.')
        if df['label'][fichier] == 1 :
             st.success ('La prévision est **correcte**.')
        else :
             st.error('La prévision est **erronée**.')
      else : 
        st.write('Le modèle prédit que ce son est **normal**.')
        if df['label'][fichier] == 0 :
             st.success ('La prévision est **correcte**.')
        else :
             st.error('La prévision est **erronée**.')
      
      if st.checkbox("Afficher le MFEC spectrogramme et écouter le son"):
         audio, fe =fc.load_audio(df1)
         st.audio(df1, format='audio/wav', start_time=0)
    
         plt.figure(figsize=(6,3))
         fig1=fc.plot_logMelEnergies(audio, fe)
         st.pyplot(fig1)
                
      
      if st.checkbox("Afficher l'évaluation de notre modèle pour "+ page):
        st.image('./Images/'+page+'_AUC.png', width=450)        
     
     
def main():
    st.sidebar.header('Détection de son anormal dans les pièces industrielles')
    page = st.sidebar.selectbox( "Menu", ["Datasets", "Analyse exploratoire", "Modélisation", "Prédiction"])
    st.sidebar.info("Projet DataScientest - Promotion Bootcamp Nov. 2020\n\n"
                 "Participants :\n"
                "[Marwan AJEM](https://www.linkedin.com/in/marwan-ajem/)\n"
                "[Nathalie BACH](https://www.linkedin.com/in/nathalie-bach-b03145196/)\n"
                "[Sara FATEN DIAZ](https://www.linkedin.com/in/sarafatendiaz/)\n\n"
                "Voir la keynote : [pdf](https://drive.google.com/file/d/1KA1pYpiWdMhygwxALc3qgkppo5B2L4yx/view)"
                )
    
    if page == "Datasets":
        dataset()
    elif page == "Analyse exploratoire":
        visualisation()
    elif page == "Modélisation":
        modelisation()
    elif page == "Prédiction":
        prédiction()
   
                    

if __name__ == "__main__":
    main()    
