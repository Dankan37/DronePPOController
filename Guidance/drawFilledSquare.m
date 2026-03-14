function Zout = drawFilledSquare(X, Y, Z, X0, Y0, X1, Y1, H)
% drawFilledSquare  Set height H inside square region.
%
% Square corners: (X0,Y0) to (X1,Y1)

mask = (X >= min(X0,X1)) & (X <= max(X0,X1)) & ...
       (Y >= min(Y0,Y1)) & (Y <= max(Y0,Y1));

Zout = Z;
Zout(mask) = H;
end
