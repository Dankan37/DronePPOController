function slopeMap = makeSlopeMap(patch, knownMask, res)
% SLOPE MAP (single channel)
%   patch     : 64x64 terrain heights (NaN = unknown)
%   knownMask : logical mask of known cells
%   res       : map resolution (meters per cell)
%
% Output:
%   slopeMap  : 64x64 matrix in [0, 1]
%               0 = flat
%               1 = very steep (clipped)

    slopeMap = zeros(size(patch));

    % If nothing is known, return zeros
    if ~any(knownMask(:))
        return
    end

    % Fill unknowns to avoid gradient explosions
    patchFilled = patch;
    patchFilled(~knownMask) = mean(patch(knownMask), 'omitnan');
    

    % Compute gradients (dz/dx, dz/dy)
    [dzdx, dzdy] = gradient(patchFilled, res);

    % Slope magnitude (meters per meter)
    slopeMag = sqrt(dzdx.^2 + dzdy.^2);

    % Normalize
    % Typical terrain slopes are < 1, cliffs >> 1
    slopeNorm = slopeMag / 2.0;     % 2 = very steep
    slopeNorm = min(max(slopeNorm, 0), 1);

    % Mask unknown areas
    slopeNorm(~knownMask) = 0;

    slopeMap = slopeNorm;
end
