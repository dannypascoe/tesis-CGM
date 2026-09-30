function Qg = ohio_gut_sim(P_mg, D0_mg, Ts_min, subj)
%OHIO_GUT_SIM  Qgut del modelo gastrointestinal (mismas ecuaciones y RK4 de
%   build_ohio_meal_signal.m) para vectores numericos, sin datetime. Sirve para
%   pruebas: permite simular comidas sinteticas y compararlas con el kernel.
%     P_mg(k)  - masa ingerida en la muestra k [mg]
%     D0_mg(k) - tamano de comida activo en la muestra k [mg] (se mantiene)
%   Devuelve Qg(k) [mg], sin calibrar.
    nT = numel(P_mg);
    Qg = zeros(nT,1);
    x  = [0; 0; 0];
    nsub = 10;  h = Ts_min / nsub;
    for kk = 2:nT
        Dt = P_mg(kk-1) / Ts_min;
        D0 = D0_mg(kk-1);
        for s = 1:nsub
            x = rk4_step(@(xx) gut_ode(xx, Dt, D0, subj), x, h);
        end
        Qg(kk) = x(3);
    end
end

function dx = gut_ode(x, Dt, D0, subj)
    Qsto1 = x(1); Qsto2 = x(2); Qgut = x(3);
    Qsto  = Qsto1 + Qsto2;
    a = 5 / (2*D0*(1-subj.b_gut));
    c = 5 / (2*D0*subj.d_gut);
    kempt = subj.kmin_gut + (subj.kmax_gut - subj.kmin_gut)/2 * ...
            ( tanh(a*(Qsto - subj.b_gut*D0)) - tanh(c*(Qsto - subj.d_gut*D0)) + 2 );
    dx = [-subj.kgri_gut*Qsto1 + Dt; ...
          -kempt*Qsto2 + subj.kgri_gut*Qsto1; ...
          -subj.kabs_gut*Qgut + kempt*Qsto2];
end

function x_next = rk4_step(f, x, h)
    k1 = f(x);
    k2 = f(x + h/2*k1);
    k3 = f(x + h/2*k2);
    k4 = f(x + h*k3);
    x_next = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
