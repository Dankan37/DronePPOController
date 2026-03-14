function [e, u, dt, log] = runFullClosedLoopSimulation(K, env, State)

    dt = env.dt;

    % References
    commanded_angles = deg2rad([10; 10; 5]);   % [phi; theta; psi] in radians
    commanded_alt    = 15;
    com_arr = [commanded_angles; commanded_alt];

    % Steps
    N = 1000;

    clear evalPid_param

    % Prealloc errors
    e.theta = zeros(N,1);
    e.phi   = zeros(N,1);
    e.psi   = zeros(N,1);
    e.z     = zeros(N,1);

    % Prealloc commands 
    u.theta = zeros(N,1);
    u.phi   = zeros(N,1);
    u.psi   = zeros(N,1);
    u.z     = zeros(N,1);

    % ---- LOGS ----
    log.t        = (0:N-1)' * dt;
    log.ang      = zeros(N,3);   % [phi theta psi] from State.sim_ang
    log.pos      = zeros(N,3);   % from State.sim_pos
    log.action   = zeros(N, numel(env.sign_x)); % int_act length (assumes vector)
    log.com_ang  = commanded_angles(:)';
    log.com_alt  = commanded_alt;
    log.rpm_raw    = zeros(N,6);
    log.rpm_smooth = zeros(N,6);


    for k = 1:N
        % PID -> actuator commands
        int_act = evalPid_param(State, env, com_arr, K);

        % Step dynamics
        State = env.fcn_step(State, int_act, env);

        % Log state
        log.ang(k,:)    = State.sim_ang(:).';
        log.pos(k,:)    = State.sim_pos(:).';
        log.action(k,:) = int_act(:).';
        log.rpm_raw(k,:)    = State.new_rpm(:).';
        log.rpm_smooth(k,:) = State.sim_rpm(:).';

        % Errors
        e.phi(k)   = commanded_angles(1) - State.sim_ang(1);
        e.theta(k) = commanded_angles(2) - State.sim_ang(2);
        e.psi(k)   = commanded_angles(3) - State.sim_ang(3);
        e.z(k)     = commanded_alt      - State.sim_pos(3);

        u.psi(k)   = abs(dot(int_act, env.arm_sign_z'));
        u.z(k)     = mean(int_act);
        u.theta(k) = abs(dot(int_act, env.sign_x));
        u.phi(k)   = abs(dot(int_act, env.sign_y));
    end
end
