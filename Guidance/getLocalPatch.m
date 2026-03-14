function [patch, knownMask] = getLocalPatch(env, pos, yaw, res)

    sz  = env.patchSize;

    % Local grid (forward = +X, left = +Y)
    half = sz/2;
    [yL, xL] = meshgrid(half:-1:(-half+1), -half:(half-1));
    xL = xL * res;
    yL = yL * res;

    % Rotation: local → world    
    fwd  = [sind(yaw), cosd(yaw)];
    left = [-cosd(yaw), sind(yaw)];

    xW = pos(1) + fwd(1).*xL + left(1).*yL;
    yW = pos(2) + fwd(2).*xL + left(2).*yL;

    % Convert → map indices
    ix = round((xW - env.XY_bound.xmin) / res) + 1;
    iy = round((yW - env.XY_bound.ymin) / res) + 1;

    patch     = nan(sz, sz);
    knownMask = false(sz, sz);

    H = size(env.heightmap,1);
    W = size(env.heightmap,2);

    valid = ix >= 1 & ix <= W & iy >= 1 & iy <= H;

    mapIdx = sub2ind([H W], iy(valid), ix(valid));
    patch(valid) = env.heightmap(mapIdx);

    knownIdx = sub2ind([H W], iy(valid), ix(valid));
    knownMask(valid) = env.height_known(knownIdx);
end
