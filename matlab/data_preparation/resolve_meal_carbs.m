function [ts, grams] = resolve_meal_carbs(meal, bolus)
%RESOLVE_MEAL_CARBS  meal.carbs como fuente principal. Ademas:
%   (a) si una comida reportada trae carbs=0, se completa con el
%       bwz_carb_input del bolo mas cercano (<=10 min);
%   (b) si un bolo trae bwz_carb_input>0 sin ninguna comida cercana
%       reportada, se agrega como un evento de comida adicional
%       (bolo dado por alimento sin registro explicito de <meal>).
    ts    = meal.ts(:);
    grams = meal.carbs(:);

    has_bolus_carbs = ~isempty(bolus.ts_begin) && isfield(bolus,'bwz_carb_input');

    if has_bolus_carbs
        for i = 1:numel(bolus.ts_begin)
            c = bolus.bwz_carb_input(i);
            if isnan(c) || c <= 0, continue; end
            if isempty(ts)
                near = false;
            else
                near = any(abs(ts - bolus.ts_begin(i)) <= minutes(10) & grams > 0);
            end
            if ~near
                ts(end+1,1)    = bolus.ts_begin(i);   %#ok<AGROW>
                grams(end+1,1) = c;                    %#ok<AGROW>
            end
        end
    end

    idx0 = find(grams == 0);
    if has_bolus_carbs
        for ii = idx0'
            [dt, j] = min(abs(bolus.ts_begin - ts(ii)));
            if dt <= minutes(10) && bolus.bwz_carb_input(j) > 0
                grams(ii) = bolus.bwz_carb_input(j);
            end
        end
    end

    [ts, order] = sort(ts);
    grams = grams(order);
end
