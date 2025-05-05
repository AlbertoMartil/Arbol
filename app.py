import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title='Clasificador de Atletas', layout='wide', initial_sidebar_state='expanded')

# Función para cargar los datos
def carga_datos():
    df = pd.read_csv('atletas.csv')
    df['Atleta'] = df['Atleta'].map({'Fondista': 1, 'Velocista': 0})
    return df

def detectar_outlier(df):
    Q1 = df[['Edad', 'Peso', 'Volumen_O2_max']].quantile(0.25)
    Q3 = df[['Edad', 'Peso', 'Volumen_O2_max']].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[['Edad', 'Peso', 'Volumen_O2_max']] < (Q1 - 1.5 * IQR)) | (df[['Edad', 'Peso', 'Volumen_O2_max']] > (Q3 + 1.5 * IQR)))
    num_outliers = outliers.sum()
    outliers_total=num_outliers.sum()
    no_outlier = ~((df[['Edad', 'Peso', 'Volumen_O2_max']] < (Q1 - 1.5 * IQR)) |(df[['Edad', 'Peso', 'Volumen_O2_max']] > (Q3 + 1.5 * IQR)))     
    df_limpio=df[no_outlier]
    outliers_2= ((df_limpio[['Edad', 'Peso', 'Volumen_O2_max']] < (Q1 - 1.5 * IQR)) | (df_limpio[['Edad', 'Peso', 'Volumen_O2_max']] > (Q3 + 1.5 * IQR)))
    num_outliers_2=outliers_2.sum()
    outliers_2_total=num_outliers_2.sum()
    return df_limpio, outliers_total, outliers_2_total

# Sidebar para parámetros del modelo y datos de usuario
def add_sidebar(df_limpio):
    st.sidebar.header('Modifica los datos')
    
    st.sidebar.title('Parámetros del modelo')
    max_depth = st.sidebar.slider('Profundidad máxima del árbol', 2, 4, 3)
    criterion = st.sidebar.selectbox('Criterio de división', ['gini', 'entropy'])
    
    st.sidebar.subheader('Modificar Variables Independientes')
    edad = st.sidebar.slider('Edad', int(df_limpio['Edad'].min()), int(df_limpio['Edad'].max()), int(df_limpio['Edad'].mean()))
    peso = st.sidebar.slider('Peso', int(df_limpio['Peso'].min()), int(df_limpio['Peso'].max()), int(df_limpio['Peso'].mean()))
    volumen_o2 = st.sidebar.slider('Volumen_O2_max', float(df_limpio['Volumen_O2_max'].min()), float(df_limpio['Volumen_O2_max'].max()), float(df_limpio['Volumen_O2_max'].mean()))
    
    return max_depth, criterion, edad, peso, volumen_o2

# Función para entrenar el modelo
def entrena_modelo(df, df_limpio, max_depth, criterion):
    X = df_limpio[['Edad', 'Peso', 'Volumen_O2_max']]
    y = df['Atleta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Selección de página
def main():
    df = carga_datos()
    df_limpio, outliers_total, outliers_2_total = detectar_outlier(df)
    max_depth, criterion, edad, peso, volumen_o2 = add_sidebar(df_limpio)
    model, X_test, y_test = entrena_modelo(df, df_limpio, max_depth, criterion)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    pagina = st.sidebar.radio("Selecciona una página:", ['Métricas y Predicción', 'Gráficos'])
    
    if pagina == 'Métricas y Predicción':
        st.title('Métricas del Modelo, Outliers y Predicción')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader('Métricas del Modelo')
            st.write(f'Precisión del modelo: {accuracy:.2f}')
            st.text(classification_report(y_test, y_pred))
            st.subheader('Outliers')
            st.write(f'El DtaFrame presenta {outliers_total} outliers, que eliminaremos con el preprocesamiento')
            st.write(f'Tras el preprocesamiento, el número de outliers es {outliers_2_total}')
            


        with col2:
            st.subheader('Predicción del Modelo')
            datos_usuario = pd.DataFrame([[edad, peso, volumen_o2]], columns=['Edad', 'Peso', 'Volumen_O2_max'])
            prediccion = model.predict(datos_usuario)[0]
            clase_predicha = 'Fondista' if prediccion == 1 else 'Velocista'
            st.write(f'Según el modelo, el atleta es un: **{clase_predicha}**')

    elif pagina == 'Gráficos':
        st.title('Gráficos del Modelo')
    
        col1, col2 = st.columns(2)
    
    
        with col1:
            st.subheader('Matriz de Confusión')
            fig, ax = plt.subplots(figsize=(5, 3))
            conf_matrix = confusion_matrix(y_test, y_pred)
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', ax=ax)
            st.pyplot(fig)

            st.subheader('Curva ROC-AUC: Comparación')

            X= df[['Edad', 'Peso', 'Volumen_O2_max']]
            y= df['Atleta']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train = X_train.dropna()
            y_train = y_train.loc[X_train.index]
            X_test = X_test.dropna()
            y_test = y_test.loc[X_test.index]

            y_pred_proba_tree = model.predict_proba(X_test)[:, 1]
            fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_proba_tree)
            auc_tree = auc(fpr_tree, tpr_tree)

            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            y_pred_proba_log = logreg.predict_proba(X_test)[:, 1]
            fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
            auc_log = auc(fpr_log, tpr_log)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC = {auc_tree:.2f})', color='blue')
            ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})', color='green')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Curva ROC - Árbol vs Regresión')
            ax.legend(loc='upper left')
            st.pyplot(fig)

            st.subheader('Árbol de Decisión')
            fig, ax = plt.subplots(figsize=(6, 4))
            
            plot_tree(model, filled=True, feature_names=['Edad', 'Peso', 'Volumen_O2_max'], class_names=['Velocista', 'Fondista'], ax=ax)
            st.pyplot(fig)

    
        with col2:
            st.subheader('Distribución de Variables')

            fig, axs = plt.subplots(3, 1, figsize=(6, 8))
            sns.histplot(df_limpio['Edad'], kde=True, ax=axs[0], color='skyblue')
            axs[0].set_title('Distribución de Edad')

            sns.histplot(df_limpio['Peso'], kde=True, ax=axs[1], color='lightgreen')
            axs[1].set_title('Distribución de Peso')

            sns.histplot(df_limpio['Volumen_O2_max'], kde=True, ax=axs[2], color='salmon')
            axs[2].set_title('Distribución de Volumen_O2_max')
            plt.tight_layout()
            st.pyplot(fig)

            

if __name__ == '__main__':
    main()
