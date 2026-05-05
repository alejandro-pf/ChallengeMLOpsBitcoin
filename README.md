# MLOps Challenge

## Introducción

Para la realización de este challenge he escogido el DataSet “Bitcoin Historical Data”, que contiene actualizaciones minuto a minuto, desde 2021, del precio de Apertura (Open), Alto (High), Bajo (Low) y Cierre (Close), junto con el volumen de ventas, del Bitcoin.

El DataSet puede encontrarse aquí:
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data?resource=download

Mi objetivo para este ejercicio será predecir el precio de cierre.

## Pasos previos y preparación 
Tras un análisis rápido, que también he subido a GitHub, se puede ver que el DataSet contiene 7530892 filas, correspondientes a todos aquellos minutos en los que se ha registrado el precio del Bitcoin. No tiene nulos y la fecha tiene formato Unix. Aunque los primeros datos parecen ser redundantes, pues hay poca variación, volumen y movimiento, según avanza el tiempo se pueden ver variaciones muy interesantes en los datos, una vez que la criptomoneda comienza a tomar relevancia pública.

Para crear el flujo de trabajo me he apoyado en el readme.md que utilizamos en el ejercicio de clase. De esta manera he configurado git, dvc y las automatizaciones de MLFlow.

Del mismo modo he creado los scripts de carga y procesado de datos y de entrenamiento.

## Creación del modelo
He decidido hacer un Random Forest como modelo de predicción, porque aunque son datos lineales, el Bitcoin no se comporta como una acción normal, por lo que ARIMA o Prophet no creo que sean los modelos más apropiados. Además, los intervalos son por minuto, no por día, por lo que creo que complicaría mucho el proceso. También he pensado usar redes neuronales, pero me parece que este es el compromiso justo.

Para el entrenamiento, he creado una fila más, que será el target y que funciona como un “lag”. Le doy a la fila el valor de cierre siguiente, que es lo que vamos a querer predecir, y así el modelo entrena a predecir el minuto siguiente.

Tal vez, como ejemplo práctico, sería más útil prever 5 minutos después, o hacer una media de los precios y saber si va a subir o a bajar, pero como ejemplo para el challenge creo que es una base correcta desde la que partir.

He separado los datos entre train y test escogiendo un 80% de los datos para train, manteniendo el orden cronológico. Sin embargo, sospecho que el modelo no va a ser demasiado bueno, pues en los años más recientes el precio del Bitcoin ha resultado ser muy volátil e impredecible.

Para registrar los datos de MLFlow he decidido desactivar el autologging y manualmente guardar el número de estimadores, la profundidad y el rmse y mae:
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

Además, para la primera ejecución he puesto 2 versiones del modelo seguidas en el archivo train, uno con 50 estimadores y profundidad de 5 y otro con 100 estimadores y profundidad de 10.

He terminado de crear el pipeline y, después de ajustar varias líneas de código que me daban error, he puesto a entrenar los modelos.

## Resultados y MLFlow
Los resultados de los 2 modelos entrenados inicialmente son los siguientes:

    RandomForest_est_50_depth_5:
      RMSE: 24503.36040750952
      MAE: 16371.673659324268
    
    RandomForest_est_100_depth_10
      RMSE: 22019.246921071302
      MAE: 13865.72605483148

Gracias a MLFlow de un vistazo se pueden ver todas las características del modelo, tanto en profundidad y estimadores como las métricas resultantes del entrenamiento, y se puede guiar la decisión de los futuros modelos en base a esto. Mi siguiente opción, después de ver los resultados y el tiempo de entrenamiento, sería mantener el número de estimadores pero aumentar la profundidad.

## Puesta en producción
Para poner el modelo en producción he usado FastAPI con la interfaz de SwaggerUI, lanzándolo con uvicorn:
 
    uvicorn src.app:app --reload
 
De esta manera he podido introducir los datos de Open, High, Low, Close y Volume de un minuto para prever el siguiente. En las pruebas que he hecho, la desviación cuadra con los errores del modelo, pero por lo menos acierta (en mis intentos) si va a subir o va a bajar. Lo cual, en mi opinión, es una buena señal.

## Mejoras a futuro y consideraciones
Me he visto algo limitado por las características de mi equipo a la hora de realizar el entrenamiento de los modelos. Pensé que iba a ser algo más rápido, pero con un DataSet tan grande mi ordenador lo ha pasado un poco mal, así que no he podido realizar todas las ejecuciones que me habría gustado para familiarizarme más con MLFlow ni “dejar que guíe mi proceso de entrenamiento”, aunque me ha resultado muy cómodo para ver las métricas y tenerlas presentes en todo momento.

Por otra parte, me ha gustado mucho el uso de GitHub. No lo había utilizado nunca para subir mis trabajos, hacer commits o llevar un control de versiones, y es algo que he notado mucho al hacer esta práctica, pero que espero implementar en el futuro.

Al no estar ni acostumbrado ni familiarizado con este tipo de trabajo, creo que me ha quedado un repositorio un poco desordenado, especialmente con los comentarios en cada commit, pero me ha resultado una herramienta muy útil e interesante.


Este trabajo se puede encontrar en el repositorio de GitHub que he creado para el proyecto:

https://github.com/alejandro-pf/ChallengeMLOpsBitcoin/tree/main

