function [tau, info] = compute_adaptive_threshold(g_ref, past_g, params)
%COMPUTE_ADAPTIVE_THRESHOLD Umbral adaptativo basado en contexto glucémico
%
%   [tau, info] = compute_adaptive_threshold(g_ref, past_g, params)
%
%   Implementa la fórmula (enfoque B+ROC):
%       τ(k) = max(τ_min, β · g_ref(k)) · (1 + γ · |ROC(k)|)
%
%   Componentes:
%       - Base proporcional (β · g_ref): alineado con ISO 15197 y MARD.
%         En hipoglucemia el umbral se reduce → mayor sensibilidad.
%         En hiperglucemia se relaja → evita falsos positivos.
%       - Factor ROC: evita confundir excursiones glucémicas reales
%         (posprandiales, correcciones insulina) con fallos del sensor.
%         Cuando la glucosa cambia rápido de forma fisiológica, el error
%         del modelo es naturalmente mayor; el factor ROC compensa esto.
%       - τ_min: piso de seguridad para valores de glucosa muy bajos.
%
%   ROC se calcula como abs(mean(diff(past_g))), que es sensible a
%   tendencias consistentes pero robusto a ruido aleatorio (donde las
%   diferencias se cancelan entre sí).
%
%   Entradas:
%       g_ref  - Valor de referencia de glucosa (predicción del modelo ŷ(k),
%                en mg/dL). Se usa la predicción y NO la lectura del sensor
%                porque esta última podría estar corrupta.
%       past_g - Vector con los últimos N valores de glucosa de la ventana
%                histórica (ya limpio, con valores <10 reemplazados).
%       params - Estructura con campos:
%                .beta    - Proporción del nivel de glucosa (default: 0.10)
%                .tau_min - Umbral mínimo en mg/dL (default: 7.0)
%                .gamma   - Peso de la tasa de cambio ROC (default: 0.05)
%
%   Salidas:
%       tau  - Umbral adaptativo calculado (mg/dL)
%       info - (Opcional) Estructura con componentes para análisis:
%              .tau_base   - Componente base proporcional (mg/dL)
%              .roc        - Tasa de cambio calculada (mg/dL por muestra)
%              .roc_factor - Factor multiplicativo aplicado por ROC
%
%   Ejemplos de comportamiento (con valores default):
%       Glucosa estable a 120 mg/dL, ROC≈0:  τ = 12.0 mg/dL
%       Glucosa estable a  70 mg/dL, ROC≈0:  τ =  7.0 mg/dL (piso)
%       Glucosa estable a 200 mg/dL, ROC≈0:  τ = 20.0 mg/dL
%       Glucosa bajando rápido (ROC=3):       τ se multiplica × 1.45
%       Glucosa subiendo rápido (ROC=4):      τ se multiplica × 1.60
%
%   Referencia: ISO 15197:2013 define precisión CGM como ±15 mg/dL
%   (glucosa <100) o ±15% (glucosa ≥100).
%
%   Preparado para extensión futura con término de varianza (Opción C):
%       τ(k) = max(τ_min, β·g_ref) · (1+γ·|ROC|) · (1+δ·σ_local)
%
%   Autor: Daniel A. Pascoe González
%   Fecha: 2026
%   Proyecto: Extensión de tesis - Umbral adaptativo para CGM

    % ===================== VALORES POR DEFECTO =====================
    if ~isfield(params, 'beta'),    params.beta    = 0.10; end
    if ~isfield(params, 'tau_min'), params.tau_min = 7.0;  end
    if ~isfield(params, 'gamma'),   params.gamma   = 0.05; end

    % ===================== COMPONENTE BASE =====================
    % Proporcional al nivel de glucosa predicho.
    % Se usa abs(g_ref) como protección contra valores negativos espurios.
    tau_base = max(params.tau_min, params.beta * abs(g_ref));

    % ===================== COMPONENTE ROC =====================
    % Tasa de cambio de glucosa en la ventana histórica.
    %
    % Se calcula como abs(mean(diff(past_g))) en lugar de mean(abs(diff)):
    %   - Tendencia real (bajada/subida sostenida): diffs son consistentes
    %     en signo → abs(mean) da valor ALTO → relaja umbral (correcto)
    %   - Ruido aleatorio: diffs positivos y negativos se cancelan
    %     → abs(mean) da valor BAJO → NO relaja umbral (correcto)
    %
    % Esto evita que el ruido relaje el umbral y solo lo hace ante
    % excursiones glucémicas reales.
    if numel(past_g) >= 2
        roc = abs(mean(diff(past_g)));
    else
        roc = 0;
    end

    roc_factor = 1 + params.gamma * roc;

    % ===================== UMBRAL FINAL =====================
    tau = tau_base * roc_factor;

    % ===================== INFO PARA ANÁLISIS =====================
    if nargout > 1
        info.tau_base   = tau_base;
        info.roc        = roc;
        info.roc_factor = roc_factor;
    end

end
