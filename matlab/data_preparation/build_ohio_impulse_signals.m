function [u_U, m_g] = build_ohio_impulse_signals(grid_ts, bolus, meal)
%BUILD_OHIO_IMPULSE_SIGNALS  Entradas "crudas" de Ohio (sin forma fisiologica).
%   Referencia inferior del experimento: cada evento se coloca como un impulso
%   en la muestra de la rejilla mas cercana.
%     u_U : dosis de bolo [U]         (sin basal ni temp_basal)
%     m_g : carbohidratos [g]         (misma resolucion de comidas que los
%                                      demas brazos: resolve_meal_carbs)
%   No se aplica calibracion ni conversion de unidades: al normalizar despues
%   con media/std de Sorensen, esta senal queda fuera de escala a proposito.
    N = numel(grid_ts);
    u_U = zeros(N,1);
    m_g = zeros(N,1);

    if ~isempty(bolus.ts_begin)
        for i = 1:numel(bolus.ts_begin)
            if isnan(bolus.dose(i)) || bolus.dose(i) <= 0, continue; end
            [~, k] = min(abs(grid_ts - bolus.ts_begin(i)));
            u_U(k) = u_U(k) + bolus.dose(i);
        end
    end

    [meal_ts, meal_g] = resolve_meal_carbs(meal, bolus);
    for i = 1:numel(meal_ts)
        if meal_g(i) <= 0, continue; end
        [~, k] = min(abs(grid_ts - meal_ts(i)));
        m_g(k) = m_g(k) + meal_g(i);
    end
end
