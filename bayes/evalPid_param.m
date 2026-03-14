function Action = evalPid_param(State, env, com_arr, K)

    %% Persistent  (integratori)
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

    %% Comandi
    com_ang = com_arr(1:3);   % [phi, theta, psi] (come nel tuo codice)
    com_alt = com_arr(4);

    %% Reset integratore quota se cambia riferimento
    if com_alt ~= lastAlt
        integ_z = 0;
    end

    %% Errori
    err_ang = com_ang - State.sim_ang;
    err_z   = com_alt - State.sim_pos(3);

    %% Inizializzazione comando
    Action = zeros(6,1);

    %% ===================== YAW (psi) =====================
    Kp = K.psi(1); Ki = K.psi(2); Kd = K.psi(3);

    integ_psi = integ_psi + err_ang(3) * env.dt;
    integ_psi = env.fcn_clamp(integ_psi, -1, 1);

    u_psi = ...
        Kp * err_ang(3) ...
      + Ki * integ_psi ...
      - Kd * State.sim_vang(3);

    u_psi = env.fcn_clamp(u_psi, -0.1, 0.1);
    Action = Action + u_psi * env.arm_sign_z';

    %% ===================== QUOTA (z) =====================
    Kp = K.z(1); Ki = K.z(2); Kd = K.z(3);

    integ_z = integ_z + err_z * env.dt;
    integ_z = env.fcn_clamp(integ_z, -1, 1);

    u_z = ...
        Kp * err_z ...
      + Ki * integ_z ...
      - Kd * State.sim_vel(3);

    u_z = env.fcn_clamp(u_z, 0.2, 0.75);
    Action = Action + u_z * ones(6,1);

    lastAlt = com_alt;

    %% ===================== ASSE X (theta) =====================
    % theta → asse X
    Kp = K.theta(1); Ki = K.theta(2); Kd = K.theta(3);

    integ_theta = integ_theta + err_ang(2) * env.dt;
    integ_theta = env.fcn_clamp(integ_theta, -1, 1);

    u_x = ...
        Kp * err_ang(2) ...
      + Ki * integ_theta ...
      - Kd * State.sim_vang(2);

    u_x = env.fcn_clamp(u_x, -0.1, 0.1);
    Action = Action + u_x * env.sign_x;

    %% ===================== ASSE Y (phi) =====================
    % phi → asse Y
    Kp = K.phi(1); Ki = K.phi(2); Kd = K.phi(3);

    integ_phi = integ_phi + err_ang(1) * env.dt;
    integ_phi = env.fcn_clamp(integ_phi, -1, 1);

    u_y = ...
        Kp * err_ang(1) ...
      + Ki * integ_phi ...
      - Kd * State.sim_vang(1);

    u_y = env.fcn_clamp(u_y, -0.1, 0.1);
    Action = Action + u_y * env.sign_y;

    %% Saturazione finale attuatori
    Action = env.fcn_clamp(Action, zeros(6,1), ones(6,1));

end
