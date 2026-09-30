function IIR = build_ohio_iir_signal(grid_ts, basal, temp_basal, bolus, Ts_min)
%BUILD_OHIO_IIR_SIGNAL Reconstruye la tasa de infusion de insulina [pmol/min]
%   sobre la rejilla: basal+temp_basal como tasa tipo escalon, bolus como
%   impulso concentrado en la muestra de rejilla mas cercana al evento.
    N = numel(grid_ts);
    rate_U_h = zeros(N,1);

    for k = 1:N
        t = grid_ts(k);
        idx = find(basal.ts <= t, 1, 'last');
        if isempty(idx)
            r = 0;
        else
            r = basal.value(idx);
        end
        tb_idx = find(temp_basal.ts_begin <= t & temp_basal.ts_end >= t, 1, 'last');
        if ~isempty(tb_idx)
            r = temp_basal.value(tb_idx);   % temp_basal sustituye la basal normal
        end
        rate_U_h(k) = r;
    end

    IIR = rate_U_h * 6000 / 60;   % U/h -> pmol/min

    if ~isempty(bolus.ts_begin)
        for i = 1:numel(bolus.ts_begin)
            [~, k] = min(abs(grid_ts - bolus.ts_begin(i)));
            dose_pmol = bolus.dose(i) * 6000;      % U -> pmol
            IIR(k) = IIR(k) + dose_pmol / Ts_min;  % pmol/min durante esa muestra,
                                                     % de forma que IIR(k)*Ts_min = dosis total
        end
    end
end
