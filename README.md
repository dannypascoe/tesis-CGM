# CGM Glucose Project

Sistema de detección de fallos e imputación en línea de datos corruptos en sensores de monitoreo continuo de glucosa (CGM) mediante redes neuronales profundas.

## Estructura del Proyecto

```
CGM_Project/
│
├── main_training.m              # Script principal de entrenamiento
├── main_online_detection.m      # Script principal de detección/imputación
├── main_metrics.m               # Script de cálculo de métricas (TODO)
├── README.md                    # Este archivo
│
├── architectures/               # Definición de arquitecturas
│   ├── build_cnn_lstm.m         # Híbrida CNN-LSTM
│   ├── build_gru.m              # Red GRU
│   ├── build_1dcnn.m            # Convolucional 1D
│   └── build_transformer_lstm.m # Híbrida Transformer-LSTM
│
├── data_preparation/            # Preparación de datos
│   ├── load_and_prepare_data.m  # Carga archivos .mat
│   ├── normalize_data.m         # Normalización z-score
│   ├── build_dataset.m          # Construcción de ventana deslizante
│   └── split_train_val.m        # División train/val
│
├── utils/                       # Funciones auxiliares
│   ├── get_training_options.m   # Opciones de entrenamiento
│   └── get_input_shape.m        # Tipo de reshape por arquitectura
│
└── data/
    ├── raw/                     # Archivos .mat originales
    └── models/                  # Modelos entrenados y resultados
```

## Arquitecturas Disponibles

| Arquitectura | Input Shape | Descripción |
|--------------|-------------|-------------|
| `cnn_lstm` | column (8×1) | Conv1D + LSTM + Dense |
| `gru` | column (8×1) | GRU + Dense |
| `1dcnn` | row (1×8) | Conv1D + MaxPool + GlobalAvgPool + Dense |
| `transformer_lstm` | row (1×8) | Attention + LSTM + Dense |

## Uso

### 1. Preparar datos

Colocar los siguientes archivos `.mat` en `data/raw/`:
- `glucose.mat` (variable: `glucosa`)
- `glucosaerror.mat` (variable: `glucosaerror`)
- `glucose_sim.mat` (variable: `glucose_sim`)
- `insulina.mat` (variable: `insulina`)
- `meal.mat` (variable: `meal`)
- `fallas_correct.mat` (variable: `true_f`)

### 2. Entrenar modelo

```matlab
% En main_training.m, modificar:
architecture = 'cnn_lstm';  % o 'gru', '1dcnn', 'transformer_lstm'

% Ejecutar el script
main_training
```

### 3. Ejecutar detección e imputación

```matlab
% En main_online_detection.m, modificar:
architecture = 'cnn_lstm';  % misma arquitectura entrenada

% Ejecutar el script
main_online_detection
```

### 4. Calcular métricas

```matlab
% Ejecutar
main_metrics  % (TODO: por implementar)
```

## Archivos Generados

Después del entrenamiento:
- `data/models/trained_<arch>_<timestamp>.mat` - Modelo con timestamp
- `data/models/trained_<arch>_latest.mat` - Último modelo entrenado

Después de detección:
- `data/models/results_<arch>_<timestamp>.mat` - Resultados con timestamp
- `data/models/results_<arch>_latest.mat` - Últimos resultados

## Variables en Archivos de Modelo

```matlab
% trained_*.mat contiene:
nets        % Cell array con 5 redes entrenadas
valMSEs     % MSE de validación de cada red
media_g, std_g, media_u, std_u, media_m, std_m  % Parámetros de normalización
N_PAST      % Pasos históricos (6)
input_shape % 'column' o 'row'
architecture
```

```matlab
% results_*.mat contiene:
g_final_all      % Señal corregida [nModels × T]
ypred_all        % Predicciones [nModels × T]
fallas_all       % Fallos detectados [nModels × T]
error_detect_all % Error vs señal corrupta [nModels × T]
error_impute_all % Error vs señal real [nModels × T]
g_test, g_test_real, true_f
```

## Autor

Daniel Alexander Pascoe González

Universidad de Guadalajara - CUCEI

Maestría en Ciencias en Robótica e Inteligencia Artificial

## Referencias

Sanchez, O.D. et al. (2025). Online Imputation of Corrupted Glucose Sensor Data Using Deep Neural Networks and Physiological Inputs. *Algorithms*, 18, 688.
