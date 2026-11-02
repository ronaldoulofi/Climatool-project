# Proyecto de predicción de variables climáticas 

Sistema de pronóstico meteorológico y detección de anomalías desarrollado con Python y Machine Learning. El proyecto integra un flujo de ingeniería de datos para recopilar y procesar información histórica, entrenar modelos predictivos y visualizar los resultados mediante una interfaz web construida con Streamlit.

El sistema trabaja con seis ciudades y cuatro variables meteorológicas: temperatura, humedad, velocidad del viento y presión atmosférica.

---

## 1. Objetivos del proyecto

- Construir un flujo ETL para recopilar, transformar y almacenar datos meteorológicos históricos.
- Desarrollar modelos de Machine Learning para estimar variables meteorológicas.
- Comparar el rendimiento de Random Forest frente a dos modelos baseline.
- Detectar anomalías de temperatura respecto al comportamiento meteorológico habitual.
- Visualizar las condiciones actuales, los pronósticos y las anomalías mediante una interfaz web interactiva.

## 2. Tecnologías utilizadas

| Tecnología | Aplicación |
|---|---|
| Python | Lenguaje principal |
| Open-Meteo API | Fuente de datos meteorológicos |
| Pandas / NumPy | Procesamiento y transformación de datos |
| Scikit-learn | Entrenamiento y evaluación de modelos |
| Random Forest | Modelos predictivos principales |
| Regression Linear | Modelo baseline |
| Isolation Forest | Detección de anomalías, si se utiliza en la implementación |
| Supabase / PostgreSQL | Almacenamiento de datos |
| Streamlit | Interfaz web |
| Matplotlib / Plotly | Visualización de datos |

## 3. Arquitectura y estructura del proyecto

```text
Weather-Project/
│
├── models/
│
├── src/
│   ├── anomaly_mod.py
│   ├── db.py
│   ├── etl_history.py
│   ├── etl.py
│   ├── interface.py
│   ├── predict.py
│   ├── train.py
│   ├── requirements.txt
│   └── .env
│
└── .gitignore
```

**Descripción de los componentes:**

- `etl_history.py`: extracción y procesamiento de datos históricos mediante Open-Meteo.
- `etl.py`: proceso ETL complementario para la alimentación de datos meteorológicos.
- `db.py`: funciones de conexión y operaciones con Supabase.
- `train.py`: preparación de datos, entrenamiento y evaluación de modelos.
- `predict.py`: generación de pronósticos meteorológicos.
- `anomaly_mod.py`: detección de anomalías meteorológicas.
- `interface.py`: aplicación web desarrollada con Streamlit.
- `models/`: almacenamiento de los modelos entrenados.

## 4. Ingeniería de datos y proceso ETL

El proyecto utiliza Open-Meteo como fuente de información meteorológica. Se recopilaron aproximadamente **dos años de datos históricos** para construir el conjunto de datos utilizado durante el entrenamiento.

El proceso ETL comprende tres etapas:

### 4.1. Extract (Extracción)

Se obtienen registros históricos mediante la API de Open-Meteo Archive, considerando las ubicaciones geográficas de las seis ciudades seleccionadas.

### 4.2. Transform (Transformación)

Los datos se procesan para:

- Estandarizar los nombres y formatos de las variables.
- Convertir las unidades meteorológicas.
- Organizar los registros cronológicamente.
- Identificar y tratar valores faltantes.
- Preparar las observaciones para el entrenamiento de los modelos.

### 4.3. Load (Carga)

Los registros procesados se almacenan en Supabase, utilizando PostgreSQL como sistema de gestión de base de datos.

La estructura permite mantener los datos meteorológicos organizados por ciudad y fecha, facilitando su consulta y posterior utilización en los procesos de Machine Learning.

## 5. Entrenamiento de modelos de Machine Learning

Se desarrollaron **cuatro modelos Random Forest independientes**, uno para cada variable meteorológica:

1. Temperatura (`temperature_c`)
2. Humedad (`humidity_pct`)
3. Velocidad del viento (`wind_speed_kmh`)
4. Presión atmosférica (`pressure_hpa`)

Esta separación permite que cada modelo aprenda los patrones correspondientes a su variable objetivo.

### 5.1. Modelos baseline

Antes de evaluar el modelo principal, se implementaron dos modelos de referencia para establecer una comparación objetiva.

**Baseline 1: Persistencia (Lag-1)**

Utiliza el último valor meteorológico observado como estimación del siguiente valor. Es una referencia sencilla para determinar si un modelo más complejo aporta una mejora real.

**Baseline 2: Regresión lineal**

Utiliza un modelo de regresión lineal para establecer una referencia basada en relaciones lineales entre las variables predictoras y la variable objetivo.

**Modelo principal: Random Forest Regressor**

Se utiliza un modelo de ensamble basado en árboles de decisión, capaz de capturar relaciones no lineales entre las variables meteorológicas.

### 5.2. Estrategia de entrenamiento y evaluación

Se utilizó una división cronológica de los datos:

- **80 % inicial:** conjunto de entrenamiento.
- **20 % final:** conjunto de evaluación.

El conjunto de evaluación contiene observaciones posteriores que no fueron utilizadas durante el entrenamiento. Esto permite medir el rendimiento del modelo sobre datos no vistos y evita mezclar aleatoriamente observaciones futuras y pasadas.

### 5.3. Métricas de evaluación

Se utilizaron dos métricas:

- **MAE (Mean Absolute Error):** mide el error absoluto promedio entre las estimaciones y los valores reales.
- **RMSE (Root Mean Squared Error):** mide la raíz del error cuadrático medio, penalizando en mayor medida los errores grandes.

En ambas métricas, los valores más bajos representan un mejor rendimiento.

## 6. Resultados de evaluación: New York

A continuación se muestran los resultados obtenidos en la ciudad de New York, comparando los dos modelos baseline frente al modelo principal.

### 6.1. Temperatura

Variable: `temperature_c`

| Modelo | MAE (°C) | RMSE (°C) |
|---|---:|---:|
| Persistencia Lag-1 | 0.77 | 1.03 |
| Regresión lineal | 0.50 | 0.77 |
| **Random Forest** | **0.48** | **0.73** |

Random Forest presenta el menor error entre los tres modelos para la temperatura.

### 6.2. Humedad

Variable: `humidity_pct`

| Modelo | MAE (%) | RMSE (%) |
|---|---:|---:|
| Persistencia Lag-1 | 3.73 | 5.13 |
| Regresión lineal | 2.81 | 4.11 |
| **Random Forest** | **2.70** | **3.87** |

Random Forest obtiene el mejor resultado para la estimación de humedad, superando ambas referencias.

### 6.3. Velocidad del viento

Variable: `wind_speed_kmh`

| Modelo | MAE (km/h) | RMSE (km/h) |
|---|---:|---:|
| Persistencia Lag-1 | 1.51 | 2.16 |
| **Regresión lineal** | **1.45** | **2.06** |
| Random Forest | 1.49 | 2.09 |

En esta variable, la regresión lineal obtiene un rendimiento ligeramente superior al modelo principal.

### 6.4. Presión atmosférica

Variable: `pressure_hpa`

| Modelo | MAE (hPa) | RMSE (hPa) |
|---|---:|---:|
| Persistencia Lag-1 | 0.38 | 0.50 |
| **Regresión lineal** | **0.31** | **0.41** |
| Random Forest | 0.32 | 0.43 |

La regresión lineal también presenta un menor error en la estimación de presión atmosférica.

### 6.5. Interpretación de resultados

Los resultados muestran que Random Forest mejora el rendimiento de ambos modelos de referencia en temperatura y humedad. Sin embargo, la regresión lineal obtiene mejores resultados en velocidad del viento y presión atmosférica.

Esta comparación evidencia que un modelo más complejo no necesariamente ofrece mejores resultados para todas las variables meteorológicas. Por ello, se mantienen las métricas individuales para evaluar objetivamente el desempeño de cada modelo.

**Nota:** estos resultados corresponden a New York. No representan todavía el rendimiento global de las seis ciudades.

## 7. Detección de anomalías meteorológicas

El proyecto incorpora un módulo adicional para identificar comportamientos meteorológicos inusuales respecto a los patrones históricos de cada ciudad.

La detección se enfoca principalmente mediante el modelo Isolation Forest, en temperaturas significativamente superiores o inferiores a las habituales.

El objetivo es identificar situaciones como:

- Temperaturas inusualmente elevadas.
- Temperaturas considerablemente inferiores a las habituales.
- Cambios térmicos que se apartan del comportamiento histórico.

Las anomalías detectadas pueden almacenarse en Supabase para su posterior consulta y visualización en la interfaz web.

La interpretación de una anomalía es estadística: no implica necesariamente un evento meteorológico extremo ni constituye una alerta oficial.

## 8. Interfaz web con Streamlit

Se desarrolló una interfaz web para consultar los resultados del sistema de manera visual e interactiva.

Sus principales funcionalidades son:

**Selector de ciudades**

Permite seleccionar cualquiera de las seis ubicaciones disponibles.

**Resumen meteorológico semanal**

Presenta un resumen de las condiciones meteorológicas y las principales variables de la semana consultada.

**Pronóstico semanal**

Muestra las estimaciones meteorológicas correspondientes a la semana de consulta, incluyendo temperatura, humedad, viento y presión.

**Gráfico de temperatura de las últimas 24 horas**

Permite comparar la temperatura observada con la temperatura prevista.

**Detección de anomalías**

Presenta gráficamente las temperaturas que se encuentran por encima o por debajo del comportamiento habitual, facilitando la identificación de posibles anomalías.

## 9. Base de datos y configuración de Supabase

El proyecto utiliza Supabase como servicio de base de datos PostgreSQL.

Para conectar la aplicación, se necesita configurar las variables de entorno.

Crear un archivo `.env` dentro de `src/` o en la ubicación que espere el código de configuración:

```env
SUPABASE_URL=https://wzjfibvseagmyxbojthl.supabase.co
SUPABASE_KEY=sb_publishable_U0DL6lLbI9aupYYumgIkow_a1kXAkzM
```

La clave publishable de Supabase está diseñada para utilizarse en aplicaciones cliente. En este proyecto se emplea para consultas de lectura.

**Seguridad:** el acceso de lectura debe estar restringido mediante Row Level Security (RLS), políticas y permisos SQL (`GRANT`). La clave pública por sí sola no garantiza que la base de datos sea de solo lectura. No se debe conceder acceso público de escritura ni incluir claves `service_role` o secret keys en el repositorio.

El archivo `.env` debe permanecer fuera de GitHub. Se recomienda incluirlo en `.gitignore` y proporcionar un archivo `.env.example` sin credenciales reales.

## 10. Instalación y ejecución

### Requisitos previos

- Python 3.10 o superior (según compatibilidad de las dependencias).
- Git.
- Acceso a la base de datos Supabase.

### 10.1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd Weather-Project
```

### 10.2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 10.3. Instalar dependencias

```bash
python -m pip install -r src/requirements.txt
```

### 10.4. Configurar las variables de entorno

Crear el archivo `.env` con las credenciales de Supabase y comprobar que `db.py` pueda leerlo correctamente.

### 10.5. Ejecutar la interfaz

Desde la raíz del proyecto:

```bash
python -m streamlit run src/interface.py
```

La aplicación estará disponible normalmente en:

```text
http://localhost:8501
```

## 11. Alcance y consideraciones

- El sistema trabaja con seis ciudades seleccionadas, no con ubicaciones meteorológicas de todo el mundo.
- Los modelos se entrenan con información histórica de aproximadamente dos años.
- Las métricas presentadas corresponden a un periodo de evaluación independiente del entrenamiento.
- Los resultados de New York son una referencia individual y no deben generalizarse a las demás ciudades.
- La precisión de los pronósticos puede variar según la ciudad, la variable meteorológica y el horizonte de predicción.
- Los resultados dependen de la disponibilidad y calidad de los datos proporcionados por Open-Meteo.
- La detección de anomalías identifica observaciones estadísticamente inusuales y no sustituye los sistemas oficiales de alerta meteorológica.

## 12. Posibles mejoras futuras

- Evaluar y comparar el rendimiento global de los modelos en las seis ciudades.
- Realizar backtesting para distintos horizontes de pronóstico.
- Optimizar los hiperparámetros de Random Forest.
- Incorporar nuevas variables meteorológicas.
- Mejorar la detección de anomalías considerando la estacionalidad.
- Automatizar la actualización de datos y el reentrenamiento de modelos.
- Desplegar la interfaz en un servicio de alojamiento en la nube.

---

## Autor

Ronaldo Paredes - Proyecto personal de portafolio enfocado en ingeniería de datos, Machine Learning y análisis meteorológico.

**Áreas aplicadas:** Data Engineering · ETL · Machine Learning · Predictive Modeling · Anomaly Detection · Data Visualization.
