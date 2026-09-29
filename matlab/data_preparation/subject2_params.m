function subj = subject2_params()
%SUBJECT2_PARAMS  Parametros reales del "Subject 2" (comentado como
%   "Subject 5" en Glucose Abs Model1), extraidos directamente del codigo
%   fuente de Subject Parameters1.m y Glucose Abs Model1.m del Simulink
%   T1DM_Patients. Es la misma parametrizacion fisiologica usada para
%   generar los datos sinteticos de entrenamiento (Sorensen/Dalla Man).

    % --- De Subject Parameters1 (insulina, glucosa) ---
    subj.VG  = 1.8892;   % dl/kg
    subj.k1  = 0.0673;   % min^-1
    subj.k2  = 0.1968;   % min^-1
    subj.VI  = 0.0462;   % l/kg
    subj.m1  = 0.2849;   % min^-1
    subj.m2  = 0.2958;   % min^-1
    subj.m3  = 0.4274;   % min^-1
    subj.m4  = 0.1183;   % min^-1
    subj.kabs = 0.0390;  % min^-1  (absorcion intestinal -> plasma, del bloque grande;
                          %          OJO: distinto del kabs de Glucose Abs Model1, ver abajo)
    subj.f   = 0.9;
    subj.kp1 = 4.1521;
    subj.kp2 = 0.00423;
    subj.kp3 = 0.0071;
    subj.ki  = 0.0106;
    subj.Fcns = 1;
    subj.Vm0 = 5.1830;
    subj.Vmx = 0.0248;
    subj.Km0 = 240.4300;
    subj.p2U = 0.0370;
    subj.ke1 = 0.0005;
    subj.ke2 = 339;
    subj.kd  = 0.0149;   % min^-1  (absorcion subcutanea de insulina)
    subj.ka1 = 0.0038;   % min^-1
    subj.ka2 = 0.0169;   % min^-1
    subj.Ib  = 91.7870;  % pmol/l (basal)
    subj.BW  = 94.0740;  % kg

    % --- De Glucose Abs Model1, caso Subject==2 ("Subject 5" en comentarios) ---
    % NOTA: este kmax/kmin/kabs/kgri es el del modelo GASTROINTESTINAL,
    % un sub-sistema distinto al de insulina; se guardan con sufijo _gut
    % para no confundirlos con los parametros de insulina de arriba.
    subj.kmax_gut = 0.0460;  % min^-1
    subj.kmin_gut = 0.0045;  % min^-1
    subj.kabs_gut = 0.0390;  % min^-1
    subj.kgri_gut = 0.0558;  % min^-1
    subj.b_gut    = 0.7092;  % adimensional
    subj.d_gut    = 0.1849;  % adimensional
end