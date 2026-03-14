function env = scanRadar(pos, yawDeg, env)
    F = env.F;
    angles = yawDeg + (-env.scan_ang : env.scan_step : env.scan_ang);

    allPts = [];

    % --- 1. Get all scan points ---
    for ang = angles
        pts = verticalFanVisibility(F, pos, ang, env.scan_params, env.XY_bound);
        if ~isempty(pts)
            allPts = [allPts; pts];
        end
    end

    if isempty(allPts)
        return
    end

    % --- 2. world → map indices ---
    ix = round((allPts(:,1) - env.XY_bound.xmin) / env.resolution) + 1;
    iy = round((allPts(:,2) - env.XY_bound.ymin) / env.resolution) + 1;

    % --- 3. Bounds check ---
    H = size(env.heightmap,1);
    W = size(env.heightmap,2);
    valid = ix >= 1 & ix <= W & iy >= 1 & iy <= H;

    if ~any(valid)
        return
    end

    % --- 4. Write all points ---
    lin = sub2ind(size(env.heightmap), iy(valid), ix(valid));
    
    env.heightmap(lin) = allPts(valid, 3);
    env.height_known(lin) = true;
end