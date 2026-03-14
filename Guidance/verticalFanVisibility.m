function visible_pts = verticalFanVisibility(F, observer_xyz, bearing_deg, scan_params, XY_bound)
    %% Parameteri
    if ~isfield(scan_params, 'max_distance'), scan_params.max_distance = 1000; end
    if ~isfield(scan_params, 'step_size'), scan_params.step_size = 2; end
    max_dist = scan_params.max_distance;
    step     = scan_params.step_size;
    
    %% setup
    p0 = observer_xyz(:)';
    az = deg2rad(bearing_deg);
    dir = [sin(az), cos(az)]; % XY plane direction
    
    %% pos
    t = (step:step:max_dist)';
    x_line = p0(1) + t*dir(1);
    y_line = p0(2) + t*dir(2);
    
    % Clamp 
    x_line = min(max(x_line, XY_bound.xmin), XY_bound.xmax);
    y_line = min(max(y_line, XY_bound.ymin), XY_bound.ymax);
    
    %% Interpolate 
    z_ground = F(x_line, y_line);
    
    %% Alt diff
    dz = z_ground - p0(3);
    elev = atan2(dz, t);
    
    %% Visibility 
    visible_mask = elev >= cummax(elev);

    edge_mask = ...
        (x_line == XY_bound.xmin) | (x_line == XY_bound.xmax) | ...
        (y_line == XY_bound.ymin) | (y_line == XY_bound.ymax);
    
    visible_mask = visible_mask & ~edge_mask;
    
    %% Output
    visible_pts = [x_line(visible_mask), y_line(visible_mask), z_ground(visible_mask)];
end