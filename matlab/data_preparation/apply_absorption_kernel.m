function y = apply_absorption_kernel(x, w, Ts_min)
%APPLY_ABSORPTION_KERNEL  Convolucion causal de una entrada x con un kernel w.
%   y[k] = Ts_min * sum_{n>=1} w[n] * x[k-n],   condiciones iniciales en cero.
%
%   Con w normalizada (sum(w)*Ts_min = 1) una entrada constante x0 produce
%   y = x0 en regimen permanente (ganancia unitaria). El retardo minimo de
%   una muestra (n>=1) coincide con la convencion de los builders ODE.
    y = filter([0; w(:) * Ts_min], 1, x(:));
end
