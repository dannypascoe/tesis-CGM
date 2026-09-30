function [w, info] = ohio_gamma_kernel(t_peak_min, t95_min, Ts_min, horizon_min)
%OHIO_GAMMA_KERNEL  Kernel de absorcion con forma gamma, definido por su moda
%   (tiempo al pico) y su percentil 95 (cola).
%
%   La densidad gamma tiene dos parametros (forma k, escala theta). Se fijan
%   pidiendo que:   moda = (k-1)*theta = t_peak_min
%                   P(T <= t95_min) = 0.95
%   Solo usa funciones de MATLAB base (gammaincinv, gammaln, fzero); no
%   requiere Statistics Toolbox.
%
%   Entradas:
%     t_peak_min  - tiempo al pico del kernel [min]
%     t95_min     - tiempo en que se acumula el 95% de la masa [min]
%     Ts_min      - cadencia de la rejilla [min]
%     horizon_min - (opcional) longitud del kernel [min]. Default 720 (12 h).
%
%   Salidas:
%     w    - Nx1, densidad muestreada en los retardos (n-0.5)*Ts_min, n=1..N,
%            en 1/min y normalizada para que sum(w)*Ts_min = 1. El retardo de
%            media muestra reproduce la convencion de retenedor de orden cero
%            de los builders ODE (una entrada en la muestra j afecta primero a
%            la muestra j+1).
%     info - struct con k, theta, moda, t95 y media.

    if nargin < 4 || isempty(horizon_min), horizon_min = 720; end

    f    = @(k) gammaincinv(0.95, k) * t_peak_min / (k - 1) - t95_min;
    k_lo = 1.0005;
    k_hi = 60;
    if f(k_lo) * f(k_hi) > 0
        error('ohio_gamma_kernel:badParams', ...
            ['No existe una gamma con moda %.0f min y percentil 95 en %.0f min. ' ...
             'La razon t95/moda debe estar entre ~1.3 y ~1000.'], t_peak_min, t95_min);
    end
    k     = fzero(f, [k_lo k_hi]);
    theta = t_peak_min / (k - 1);

    N   = round(horizon_min / Ts_min);
    lag = ((1:N)' - 0.5) * Ts_min;
    g   = exp((k - 1)*log(lag) - lag/theta - gammaln(k) - k*log(theta));
    w   = g / (sum(g) * Ts_min);          % renormaliza tras truncar en 'horizon'

    info = struct('shape_k', k, 'scale_min', theta, 'mode_min', t_peak_min, ...
                  't95_min', t95_min, 'mean_min', k*theta, 'horizon_min', horizon_min);
end
