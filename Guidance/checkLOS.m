function [losClear, minClearance] = checkLOS(dronePos, targetPos, F)
% checkLOS: TRUE if the straight line from drone → target is not blocked
%
% Inputs:
%   dronePos  = [x;y;z]
%   targetPos = [x;y;z]
%   F         = griddedInterpolant for terrain height: h = F(x,y)
%
% Outputs:
%   losClear      : logical → true if LOS unobstructed
%   minClearance  : minimum "drone-path altitude minus terrain" along LOS

    % Horizontal distance
    vec = targetPos(1:2) - dronePos(1:2);
    distXY = norm(vec);

    if distXY < 1e-6
        losClear = true;
        minClearance = dronePos(3) - F(dronePos(1),dronePos(2));
        return;
    end

    % Direction in XY
    dirXY = vec / distXY;

    % Sample every 1 m along the path
    step = 1.0;
    N = max(1, floor(distXY / step));

    minClearance = inf;

    for i = 1:N
        % Point along the LOS in XY
        p_xy = dronePos(1:2) + dirXY * (i * step);

        % Interpolated terrain height
        h_ground = F(p_xy(1), p_xy(2));

        % Linear interpolation of altitude from drone to target
        t = (i * step) / distXY;
        h_line = dronePos(3) * (1 - t) + targetPos(3) * t;

        % Clearance
        clearance = h_line - h_ground;
        if clearance < minClearance
            minClearance = clearance;
        end

        % LOS is blocked
        if clearance < 0
            losClear = false;
            return;
        end
    end

    % No block found
    losClear = true;
end
