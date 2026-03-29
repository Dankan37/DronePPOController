%% Cascaded XY Position->Attitude PID tuning via Bayesian Optimization
% Outer PID (x,y position) -> Inner (pretrained) PID (phi,theta,psi,z)
% Altitude is PASS-THROUGH to inner z PID (no outer z controller)

clear; clc; close all;

% load env and dynamics
run("Baseline_last.mlx");
rng(1,'twister'); % only if you want determinism

% fixed inner loop gains
% Replace with your real best inner gains
load("k_in.mat")

% outer loop tuning variables
vars = [
    optimizableVariable('kp_x',[-5 5])
    optimizableVariable('ki_x',[0 2])
    optimizableVariable('kd_x',[-5 5])

    optimizableVariable('kp_y',[-5 5])
    optimizableVariable('ki_y',[0 2])
    optimizableVariable('kd_y',[-5 5])
];

% bayesian optimization
objFun = @(params) cascadedXYObjectiveWrapper(params, env, Kinner);


% best gains and final simulation

Kouter.x = [2.974, 0.001, 0.231];
Kouter.y = [3.01, 0.00078, 0.233];

[~, State0] = resetFunction(env);
[e, u, dt, log] = runCascadedXYClosedLoopSimulation(Kouter, Kinner, env, State0);

t = log.t;

figure('Name','Position');
plot(t, log.pos(:,1), t, log.pos(:,2), t, log.pos(:,3), 'LineWidth', 1.5);
grid on; xlabel('Tempo [s]'); ylabel('Posizione [m]');
legend('x','y','z','Location','best');
title('Posizione');

hold on;
yline(log.pos_ref(1), '--', 'HandleVisibility','off');
yline(log.pos_ref(2), '--', 'HandleVisibility','off');
yline(log.pos_ref(3), '--', 'HandleVisibility','off');
hold off;

figure('Name','Outer commands (phi/theta)');
plot(t, log.cmd_ang(:,1), t, log.cmd_ang(:,2), 'LineWidth', 1.5);
grid on; xlabel('Tempo [s]'); ylabel('Angolo comandato [rad]');
legend('\phi_{cmd}','\theta_{cmd}','Location','best');
title('Angoli comandati');

figure('Name','Angles (actual)');
plot(t, log.ang(:,1), t, log.ang(:,2), t, log.ang(:,3), 'LineWidth', 1.5);
xlabel('Tempo [s]'); ylabel('Angoli [rad]');
legend('\phi','\theta','\psi','Location','best');
title('Angoli');

figure('Name','Errors');
plot(t, e.x, t, e.y, t, e.z, t, e.phi, t, e.theta, 'LineWidth', 1.2);
grid on; xlabel('Time [s]'); ylabel('Errore');
legend('e_x','e_y','e_z','e_\phi','e_\theta','Location','best');
title('Errori normalizzati');

figure('Name','Actuator commands');
plot(t, log.action, 'LineWidth', 1.0);
grid on; xlabel('Tempo [s]'); ylabel('Comando motore');
title('Comandi motore');
legend("1","2","3","4","5","6","Location","bestoutside");



% objective wrapper for outer loop
function J = cascadedXYObjectiveWrapper(x, env, Kinner)

    Kouter.x = [x.kp_x, x.ki_x, x.kd_x];
    Kouter.y = [x.kp_y, x.ki_y, x.kd_y];

    % Reset persistents each objective evaluation
    clear evalPid_inner
    clear evalPid_outer_xy

    [~, State] = resetFunction(env);
    [e, u, dt, ~] = runCascadedXYClosedLoopSimulation(Kouter, Kinner, env, State);

    % Scaling
    escale = struct( ...
        'x',  5, ...
        'y',  5, ...
        'z',  5, ...                  % only for terminal/regularization if used
        'phi',   deg2rad(15), ...
        'theta', deg2rad(15) ...
    );

    % Weights
    we_pos = struct('x', 50, 'y', 50);          % main objective: x/y
    we_ang = struct('phi', 1.0, 'theta', 1.0);  % keep attitudes reasonable (optional)

    wu  = struct('phi',1e-3,'theta',1e-3,'psi',1e-3,'z',1e-3);
    wdu = struct('phi',5e-3,'theta',5e-3,'psi',5e-3,'z',5e-2);

    alpha = 0.9;
    if isfield(env,'J_u_smooth_alpha'); alpha = env.J_u_smooth_alpha; end

    J = 0;

    % Position tracking (x,y)
    ex = e.x / escale.x;
    ey = e.y / escale.y;
    J = J + (we_pos.x*sum(ex.^2) + we_pos.y*sum(ey.^2)) * dt;

    % Optional angle regularization (inner loop stays near commanded)
    ephi   = e.phi   / escale.phi;
    etheta = e.theta / escale.theta;
    J = J + (we_ang.phi*sum(ephi.^2) + we_ang.theta*sum(etheta.^2)) * dt;

    % Effort + smoothness (from your projections)
    fields = fieldnames(u);
    for i = 1:numel(fields)
        f = fields{i};
        un = u.(f);

        if alpha < 1.0
            us = zeros(size(un));
            us(1) = un(1);
            for k = 2:numel(un)
                us(k) = us(k-1) + alpha*(un(k) - us(k-1));
            end
        else
            us = un;
        end

        du = [0; diff(un)];
        J = J + (wu.(f)*sum(us.^2) + wdu.(f)*sum((du/dt).^2)) * dt;
    end

    % Terminal penalty: strongly enforce final x/y
    J = J + 100*((e.x(end)/escale.x)^2 + (e.y(end)/escale.y)^2);

    if ~isfinite(J)
        J = 1e6;
    end
end


% cascaded simulation
function [e, u, dt, log] = runCascadedXYClosedLoopSimulation(Kouter, Kinner, env, State)

    dt = env.dt;
    N  = 3000;

    % Position reference (use your target)
    pos_ref = State.sim_tgt_pos;
    if isempty(pos_ref) || numel(pos_ref) ~= 3
        pos_ref = [5; 5; 15];
    end

    % Pass-through altitude command (inner z PID tracks this)
    z_cmd = pos_ref(3);

    % Desired yaw (fixed)
    psi_cmd = deg2rad(5);

    % Outer loop command limits
    phi_max   = deg2rad(20);
    theta_max = deg2rad(20);

    % Prealloc errors
    e.x     = zeros(N,1);
    e.y     = zeros(N,1);
    e.z     = zeros(N,1);      % not optimized, but logged
    e.phi   = zeros(N,1);
    e.theta = zeros(N,1);
    e.psi   = zeros(N,1);

    % Effort projections (same as your previous code)
    u.theta = zeros(N,1);
    u.phi   = zeros(N,1);
    u.psi   = zeros(N,1);
    u.z     = zeros(N,1);

    % Logs
    log.t       = (0:N-1)' * dt;
    log.pos     = zeros(N,3);
    log.ang     = zeros(N,3);
    log.action  = zeros(N, numel(env.sign_x));
    log.pos_ref = pos_ref(:).';
    log.cmd_ang = zeros(N,3);     % [phi_cmd theta_cmd psi_cmd]
    log.cmd_z   = z_cmd * ones(N,1);

    log.rpm_raw    = zeros(N,6);
    log.rpm_smooth = zeros(N,6);

    clear evalPid_inner
    clear evalPid_outer_xy

    for k = 1:N

        % OUTER PID: x/y position -> phi/theta commands only
        cmd.pos_ref = pos_ref;
        [phi_cmd, theta_cmd] = evalPid_outer_xy(State, env, cmd, Kouter);

        phi_cmd   = env.fcn_clamp(phi_cmd,   -phi_max,   phi_max);
        theta_cmd = env.fcn_clamp(theta_cmd, -theta_max, theta_max);

        % Compose inner references [phi;theta;psi;z]
        com_arr = [phi_cmd; theta_cmd; 0; z_cmd];

        % INNER PID -> actuators
        int_act = evalPid_inner(State, env, com_arr, Kinner);

        % Step dynamics
        State = env.fcn_step(State, int_act, env);

        % Log
        log.pos(k,:)    = State.sim_pos(:).';
        log.ang(k,:)    = State.sim_ang(:).';
        log.action(k,:) = int_act(:).';
        log.cmd_ang(k,:)= [phi_cmd, theta_cmd, psi_cmd];

        if isfield(State,'new_rpm'); log.rpm_raw(k,:) = State.new_rpm(:).'; end
        if isfield(State,'sim_rpm'); log.rpm_smooth(k,:) = State.sim_rpm(:).'; end

        % Errors
        e.x(k) = pos_ref(1) - State.sim_pos(1);
        e.y(k) = pos_ref(2) - State.sim_pos(2);
        e.z(k) = pos_ref(3) - State.sim_pos(3);

        e.phi(k)   = phi_cmd   - State.sim_ang(1);
        e.theta(k) = theta_cmd - State.sim_ang(2);
        e.psi(k)   = psi_cmd   - State.sim_ang(3);

        % Effort projections
        u.psi(k)   = abs(dot(int_act, env.arm_sign_z'));
        u.z(k)     = mean(int_act);
        u.theta(k) = abs(dot(int_act, env.sign_x));
        u.phi(k)   = abs(dot(int_act, env.sign_y));
    end
end


% outer pid x y to phi theta
function [phi_cmd, theta_cmd] = evalPid_outer_xy(State, env, cmd, Kouter)

    persistent ix iy
    persistent last_pos_ref

    if isempty(ix)
        ix = 0; iy = 0;
        last_pos_ref = [NaN;NaN;NaN];
    end

    pos_ref = cmd.pos_ref(:);

    % Reset integrators if reference changes
    if any(pos_ref ~= last_pos_ref)
        ix = 0; iy = 0;
        last_pos_ref = pos_ref;
    end

    % Position errors
    ex = (pos_ref(1) - State.sim_pos(1)) / 10;
    ey = (pos_ref(2) - State.sim_pos(2)) / 10;

    % Integrate
    ix = ix + ex * env.dt;
    iy = iy + ey * env.dt;

    % Anti-windup clamp
    ix = env.fcn_clamp(ix, -1, 1);
    iy = env.fcn_clamp(iy, -1, 1);

    % Derivative from velocity
    vx = State.sim_vel(1);
    vy = State.sim_vel(2);

    % Gains
    Kpx = Kouter.x(1); Kix = Kouter.x(2); Kdx = Kouter.x(3);
    Kpy = Kouter.y(1); Kiy = Kouter.y(2); Kdy = Kouter.y(3);

    % Mapping: x -> theta_cmd, y -> phi_cmd
    % If sign is wrong in your dynamics, flip one or both signs.
    theta_cmd = (Kpx*ex + Kix*ix - Kdx*vx);
    phi_cmd   = -(Kpy*ey + Kiy*iy - Kdy*vy);
end


% inner pid controller
function Action = evalPid_inner(State, env, com_arr, K)

    persistent integ_z
    persistent integ_theta integ_phi integ_psi
    persistent lastAlt

    if isempty(integ_z)
        integ_z     = 0;
        integ_theta = 0;
        integ_phi   = 0;
        integ_psi   = 0;
        lastAlt     = 0;
    end

    com_ang = com_arr(1:3);
    com_alt = com_arr(4);

    if com_alt ~= lastAlt
        integ_z = 0;
    end

    err_ang = com_ang - State.sim_ang;
    err_z   = com_alt - State.sim_pos(3);

    Action = zeros(6,1);

    % YAW (psi)
    Kp = K.psi(1); Ki = K.psi(2); Kd = K.psi(3);
    integ_psi = integ_psi + err_ang(3) * env.dt;
    integ_psi = env.fcn_clamp(integ_psi, -1, 1);

    u_psi = Kp*err_ang(3) + Ki*integ_psi - Kd*State.sim_vang(3);
    u_psi = env.fcn_clamp(u_psi, -0.1, 0.1);
    Action = Action + u_psi * env.arm_sign_z';

    % ALTITUDE (z)
    Kp = K.z(1); Ki = K.z(2); Kd = K.z(3);
    integ_z = integ_z + err_z * env.dt;
    integ_z = env.fcn_clamp(integ_z, -1, 1);

    u_z = Kp*err_z + Ki*integ_z - Kd*State.sim_vel(3);
    u_z = env.fcn_clamp(u_z, 0.2, 0.75);
    Action = Action + u_z * ones(6,1);

    lastAlt = com_alt;

    % THETA (x axis)
    Kp = K.theta(1); Ki = K.theta(2); Kd = K.theta(3);
    integ_theta = integ_theta + err_ang(2) * env.dt;
    integ_theta = env.fcn_clamp(integ_theta, -1, 1);

    u_x = Kp*err_ang(2) + Ki*integ_theta - Kd*State.sim_vang(2);
    u_x = env.fcn_clamp(u_x, -0.1, 0.1);
    Action = Action + u_x * env.sign_x;

    % PHI (y axis)
    Kp = K.phi(1); Ki = K.phi(2); Kd = K.phi(3);
    integ_phi = integ_phi + err_ang(1) * env.dt;
    integ_phi = env.fcn_clamp(integ_phi, -1, 1);

    u_y = Kp*err_ang(1) + Ki*integ_phi - Kd*State.sim_vang(1);
    u_y = env.fcn_clamp(u_y, -0.1, 0.1);
    Action = Action + u_y * env.sign_y;

    Action = env.fcn_clamp(Action, zeros(6,1), ones(6,1));
end


% reset
function [InitialObservation,State] = resetFunction(env)

    sim_pos  = [0; 0; 10];
    sim_vel  = zeros(3,1);
    sim_ang  = zeros(3,1);
    sim_vang = zeros(3,1);
    sim_int  = zeros(3,1);

    sim_tgt_pos = [5; 10; 15];

    sim_rpm = 3000 * ones(6,1);

    err_pos = (sim_tgt_pos - sim_pos) ./ 10;
    vel_nrm = sim_vel / 5;
    ang_nrm = sim_ang / (pi/4);

    InitialObservation = [err_pos; vel_nrm; ang_nrm; sim_int];

    sim_time = 0;
    Action   = zeros(3,1);

    State = struct(...
        'sim_ang', sim_ang, ...
        'sim_vel', sim_vel, ...
        'sim_pos', sim_pos, ...
        'sim_vang', sim_vang, ...
        'sim_tgt_pos', sim_tgt_pos, ...
        'sim_time',sim_time,...
        'last_act', Action,...
        'sim_int',sim_int,...
        'last_rpm', sim_rpm ...
    );
end
