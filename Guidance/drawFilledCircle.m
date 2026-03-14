function Zout = drawFilledCircle(X, Y, Z, cx, cy, R, H)
% drawFilledCircle  Set height H inside circular region.

mask = (X - cx).^2 + (Y - cy).^2 <= R^2;

Zout = Z;
Zout(mask) = H;
end
