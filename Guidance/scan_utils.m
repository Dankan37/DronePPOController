%% Load terrain data
data = readmatrix("terrain_300.csv");
X = data(:,1); Y = data(:,2); Z = data(:,3) * 0; % Divide Z by 10

m_size = 300; nmax = 5;
k_scale = m_size * 0.8 / nmax;

for i = 1:8
    Z = drawFilledCircle(X,Y,Z, i * k_scale, i * k_scale, m_size/40, 50);
    Z = drawFilledCircle(X,Y,Z, m_size - i * k_scale, i * k_scale, m_size/40, 50);
end

Z = drawFilledCircle(X,Y,Z, m_size / 2, m_size / 2, m_size/40, 50);


ux = unique(X);
uy = unique(Y);
[Xg, Yg] = ndgrid(uy, ux);   % note: (Y, X) ordering
Zg = reshape(Z, numel(uy), numel(ux));  

F = griddedInterpolant(Xg, Yg, Zg, 'linear', 'nearest');

XY_bound.xmax = max(X(:));
XY_bound.xmin = min(X(:));
XY_bound.ymax = max(Y(:));
XY_bound.ymin = min(Y(:));

uniqueX = unique(X); uniqueY = unique(Y);
Zgrid = reshape(Z, numel(uniqueY), numel(uniqueX));
[Xgrid, Ygrid] = meshgrid(uniqueX, uniqueY);
