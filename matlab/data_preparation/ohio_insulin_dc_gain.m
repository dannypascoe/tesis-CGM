function G = ohio_insulin_dc_gain(subj)
%OHIO_INSULIN_DC_GAIN  Ganancia estatica del sub-modelo de insulina.
%   G [mU/L por pmol/min]: insulina plasmatica en regimen permanente ante una
%   tasa de infusion constante de 1 pmol/min. Sale de igualar a cero las
%   cuatro derivadas del sistema de build_ohio_insulin_signal.m:
%
%       Ip_ss = IIR / ( BW * (m4 + m2*m3/(m1+m3)) )        [pmol/kg]
%       u_ss  = Ip_ss / VI / 6                              [mU/L]
%
%   Se usa para que el kernel alternativo entregue u(k) en las mismas
%   unidades y con el mismo nivel medio que la senal de entrenamiento.
    G = 1 / ( subj.BW * subj.VI * 6 * (subj.m4 + subj.m2*subj.m3/(subj.m1 + subj.m3)) );
end
