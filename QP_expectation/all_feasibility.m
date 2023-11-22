function feas = all_feasibility(x, s, w, ep, gm)
%    [~, b] = size(w);
    [~, N] = size(ep);
    T = sum((x - s).*w, 1)' + ep;
    feas = sum(T.*T, 2)/N - gm;
%     feas = zeros(b, 1);
%     for j = 1 : b
%         feas(j) = (ep(j, :) + (x - s(:, j))' * w(:, j)) * (ep(j, :) + (x - s(:, j))' * w(:, j))' / N - gm(j);
%     end
end
