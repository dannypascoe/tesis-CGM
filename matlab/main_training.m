%% =========================================================================
%  MAIN_TRAINING.M - Script Principal de Entrenamiento
%  =========================================================================
%  Descripción: Entrena redes neuronales para detección de fallos e 
%               imputación de datos de glucosa CGM.
%  
%  Arquitecturas disponibles:
%    1. 'cnn_lstm'        - Híbrida CNN-LSTM
%    2. 'gru'             - Red GRU
%    3. '1dcnn'           - Red Convolucional 1D
%    4. 'transformer_lstm' - Híbrida Transformer-LSTM
%
%  Uso: Modificar la variable 'architecture' y ejecutar el script completo.
%  =========================================================================

clear; clc; close all;

%% ===================== AGREGAR CARPETAS AL PATH =====================
% OPCIÓN 1: Ruta manual (MODIFICAR SEGÚN TU UBICACIÓN)
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';

% OPCIÓN 2: Automática (solo funciona si ejecutas el script COMPLETO, no por secciones)
% scriptPath = fileparts(mfilename('fullpath'));

% Agregar subcarpetas al path de MATLAB
addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));
addpath(fullfile(scriptPath, 'data'));

fprintf('📁 Carpetas agregadas al path de MATLAB\n');
%% ===================== CONFIGURACIÓN =====================

% Seleccionar arquitectura a entrenar
% Opciones: 'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'
architecture = 'transformer_lstm';

% Hiperparámetros generales
N_PAST   = 6;      % Número de pasos históricos de glucosa
N_FUTURO = 1;      % Horizonte de predicción (1 paso)

% Parámetros de entrenamiento
numRuns = 5;                           % Número de entrenamientos con diferentes semillas
seeds   = [42, 100, 2024, 7, 1337];    % Semillas para reproducibilidad
valRatio = 0.2;                        % Porcentaje de datos para validación

% Ruta de datos (relativa a la ubicación del script)
data_path = fullfile(scriptPath, 'data', 'raw');

% Ruta para guardar modelos
models_path = fullfile(scriptPath, 'data', 'models');

%% ===================== CARGAR Y PREPARAR DATOS =====================

fprintf('📂 Cargando datos...\n');
[g_train, g_test, g_test_real, u, m, true_f] = load_and_prepare_data(data_path);

N = length(g_train);
fprintf('   ✅ Datos cargados: %d muestras de entrenamiento\n', N);

%% ===================== NORMALIZACIÓN =====================

fprintf('📊 Normalizando datos...\n');
[g_train_norm, u_norm, m_norm, norm_params] = normalize_data(g_train, u, m);

% Extraer parámetros de normalización (para uso posterior)
media_g = norm_params.media_g;
std_g   = norm_params.std_g;
media_u = norm_params.media_u;
std_u   = norm_params.std_u;
media_m = norm_params.media_m;
std_m   = norm_params.std_m;

fprintf('   ✅ Normalización completada\n');

%% ===================== CONSTRUIR DATASET =====================

fprintf('🔧 Construyendo conjunto de entrenamiento...\n');

% Determinar tipo de reshape según arquitectura
input_shape = get_input_shape(architecture);

% Construir dataset
[Xtrain, Ytrain] = build_dataset(g_train_norm, u_norm, m_norm, N_PAST, input_shape);

fprintf('   ✅ Dataset construido: %d muestras (shape: %s)\n', numel(Ytrain), input_shape);

%% ===================== DIVISIÓN TRAIN/VAL =====================

fprintf('📋 Dividiendo en entrenamiento y validación...\n');
[XTrainSplit, YTrainSplit, XValSplit, YValSplit, numTrain] = ...
    split_train_val(Xtrain, Ytrain, valRatio);

fprintf('   ✅ Train: %d muestras | Val: %d muestras\n', ...
    numel(YTrainSplit), numel(YValSplit));

%% ===================== DEFINIR ARQUITECTURA =====================

fprintf('🏗️  Construyendo arquitectura: %s\n', upper(architecture));

inputSize = N_PAST + 2;  % 6 glucosa + 1 insulina + 1 meal

switch architecture
    case 'cnn_lstm'
        [layers, lgraph] = build_cnn_lstm(inputSize, N_FUTURO);
        use_lgraph = false;
        
    case 'gru'
        [layers, lgraph] = build_gru(inputSize, N_FUTURO);
        use_lgraph = false;
        
    case '1dcnn'
        [layers, lgraph] = build_1dcnn(inputSize, N_FUTURO);
        use_lgraph = false;
        
    case 'transformer_lstm'
        [layers, lgraph] = build_transformer_lstm(inputSize, N_FUTURO);
        use_lgraph = true;  % Transformer requiere layerGraph
        
    otherwise
        error('Arquitectura no reconocida: %s', architecture);
end

fprintf('   ✅ Arquitectura definida\n');

%% ===================== OPCIONES DE ENTRENAMIENTO =====================

options = get_training_options(XValSplit, YValSplit, numTrain);

fprintf('   ✅ Opciones de entrenamiento configuradas\n');

%% ===================== ENTRENAMIENTO CON MÚLTIPLES SEMILLAS =====================

fprintf('\n🚀 Iniciando entrenamiento con %d semillas...\n', numRuns);
fprintf('   ─────────────────────────────────────────\n');

valMSEs = zeros(numRuns, 1);
nets    = cell(numRuns, 1);

for i = 1:numRuns
    rng(seeds(i));  % Fijar semilla
    
    fprintf('   [%d/%d] Semilla %d... ', i, numRuns, seeds(i));
    
    % Entrenar según tipo de arquitectura
    if use_lgraph
        [net, info] = trainNetwork(XTrainSplit, YTrainSplit, lgraph, options);
    else
        [net, info] = trainNetwork(XTrainSplit, YTrainSplit, layers, options);
    end
    
    % Guardar red y mejor pérdida de validación
    nets{i} = net;
    valMSEs(i) = min(info.ValidationLoss);
    
    fprintf('Val MSE = %.4f\n', valMSEs(i));
end

%% ===================== ESTADÍSTICAS DE ENTRENAMIENTO =====================

meanMSE = mean(valMSEs);
stdMSE  = std(valMSEs);

fprintf('   ─────────────────────────────────────────\n');
fprintf('📊 MSE de validación promedio: %.4f ± %.4f\n', meanMSE, stdMSE);

%% ===================== GUARDAR MODELO =====================

fprintf('\n💾 Guardando modelo...\n');

% Crear nombre de archivo con timestamp
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
model_filename = fullfile(models_path, sprintf('trained_%s_%s.mat', architecture, timestamp));

% Guardar todas las variables necesarias para imputación
save(model_filename, ...
    'nets', 'valMSEs', 'meanMSE', 'stdMSE', ...
    'media_g', 'std_g', ...
    'media_u', 'std_u', ...
    'media_m', 'std_m', ...
    'N_PAST', 'N_FUTURO', ...
    'input_shape', ...
    'architecture', ...
    'seeds', 'numRuns');

% También guardar una versión "latest" para fácil acceso
model_latest = fullfile(models_path, sprintf('trained_%s_latest.mat', architecture));
save(model_latest, ...
    'nets', 'valMSEs', 'meanMSE', 'stdMSE', ...
    'media_g', 'std_g', ...
    'media_u', 'std_u', ...
    'media_m', 'std_m', ...
    'N_PAST', 'N_FUTURO', ...
    'input_shape', ...
    'architecture', ...
    'seeds', 'numRuns');

fprintf('   ✅ Modelo guardado en: %s\n', model_filename);
fprintf('   ✅ Copia "latest" en: %s\n', model_latest);

%% ===================== RESUMEN FINAL =====================

fprintf('\n═══════════════════════════════════════════\n');
fprintf('✅ ENTRENAMIENTO COMPLETADO\n');
fprintf('   Arquitectura: %s\n', upper(architecture));
fprintf('   Semillas: %d entrenamientos\n', numRuns);
fprintf('   Val MSE: %.4f ± %.4f\n', meanMSE, stdMSE);
fprintf('═══════════════════════════════════════════\n');
