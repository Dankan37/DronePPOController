
%Drone
env.drn_mass = 15; %kg
env.drn_arm = 0.35; %m
env.drn_arm_c = 6; %num braccia

%inertia
env.inertia = [0.2; 0.2; 0.7]; %mom inerzia ixx iyy izz (x rollio, y pitch, i yaw) kg/m^2

env.in_mat = [env.inertia(1), 0,   0;
                0, env.inertia(2), 0;
                0, 0, env.inertia(3)];
env.inv_intmat = inv(env.in_mat);

%Attrito
env.torqueFric =0.5 * 1.12 * [0.2, 0, 0;
                0, 0.2, 0;
                0, 0, 0.2];

%variabili
env.arm_angle = 360 / env.drn_arm_c; %angolo tra braccia [deg]

%posizione braccia rispetto centro, disposti su una circonferenza
max_angle = 360 - env.arm_angle; %angolo massimo tra primo e ultimo braccio
env.arm_pos_x = env.drn_arm * cosd(0:env.arm_angle:max_angle);
env.arm_pos_y = env.drn_arm * sind(0:env.arm_angle:max_angle);
env.arm_sign_z = [1, -1, 1, -1, 1, -1];

%Segni per comando
env.sign_z = [1; -1; 1; -1; 1;-1];
env.sign_x = [-1; -1; 1; 1; 1; -1]; %segni spinta per mom attorno a y positivo
env.sign_y = [0; 1; 1; 0; -1; -1];

%Limiti
env.lowerlimit = [-0.5; -0.5; -0.5; -1];
env.upperlimit = [+0.5; +0.5; +0.5; 1];

%Simulazione
env.dt = 0.0025;
env.g = 9.81;
env.rho = 1.12;
env.S = 0.5;
env.cd = 15;
env.k0 = 0.5 * env.rho * env.S * env.cd;
env.rpm_smooth = 0.06;
env.plot = false;
env.sim_curr = 0;

%Rotore https://www.kdedirect.com/collections/uas-multi-rotor-brushless-motors/products/kde13218xf-105
Kv =	330; %RPM/V
Kt =	0.0289; %Nm/A
Rm =	0.044; %Ω
Ke =    1 / (Kv * (2*pi) / 60); %Vs/rad
I0 =    0.5; %A
J  =    0.631 / 100^2; %kg*m^2

%Aprox B
% run("boh.m")
% B = B_fun; %best fit dati costruttore



%Elica KDE-CF355-DP https://www.kdedirect.com/collections/multi-rotor-propeller-blades/products/kde-cf355-dp
C_thr = 0.0763 * 1;
C_tor = 0.00359 * 1;
C_pow = 0.02254;
rho = 1.11;
d = 0.5461;
m = 2 * 0.025;

env.prp_in = 1/12 * m * d^2 + J; 
xstuff = (0:100:4500);

Thr = C_thr * rho * d^4 * (xstuff./60).^2;
plot(xstuff,Thr, LineWidth=2.5)
title("Spinta vs RPM")
xlabel("RPM")
ylabel("T [N]")
grid on
Tor = C_tor * rho * d^5 * (xstuff./60).^2;
plot(xstuff,Tor, LineWidth=2.5)
title("Coppia vs RPM")
xlabel("RPM")
ylabel("M [Nm]")
grid on

%Funzioni Torque e Thrust
env.fcn_tor     = @(rpm) C_tor * rho * d^5 * (rpm/60).^2;
env.fcn_thr     = @(rpm) C_thr * rho * d^4 * (rpm/60).^2;
env.fcn_pow     = @(rpm) C_pow * rho * d^5 * (rpm/60).^3;
env.fcn_clamp   = @(x,a,b) min(b, max(a, x));

function rpm_val = throttle2rpm(throttle)
    thr = [0.25, .375, 0.5, 0.625, 0.75, 0.875, 1]';
    rpm = [2140, 2960, 3690, 4340, 4930, 5250, 5830]';

    rpm_val = interp1(thr, rpm, throttle, 'linear', 'extrap');
end
env.throttle2rpm = @(throttle) throttle2rpm(throttle);


function arr = randomArrayBound(arrMin, arrMax)
    N = length(arrMin);
    arr = zeros(N,1);
    for i=1:N
        arr(i) = arrMin(i) + rand(1) * (arrMax(i) - arrMin(i));
    end
end
env.fcn_randomArrayBound = @(a,b) randomArrayBound(a,b);


%Funzioni step eulero
function State = stepState(State, Action, env)
    % Ordina struct
    sim_ang     = State.sim_ang;
    sim_vel     = State.sim_vel;
    sim_pos     = State.sim_pos;
    sim_vang    = State.sim_vang;
    sim_time    = State.sim_time;

    throttle_val = Action;

    %RPM Smooth
    last_rpm = State.last_rpm;
    new_rpm     = env.throttle2rpm(throttle_val);
    
    %Time of sim
    sim_time = sim_time + env.dt;

    %Smooth -> Low Pass filter
    sim_rpm = last_rpm * (1 - env.rpm_smooth) + env.rpm_smooth * new_rpm;
    sim_wrot    = sim_rpm * 2 * pi / 60;
    
    %Funzioni
    thrust_arr = env.fcn_thr(sim_rpm);
    torque_arr = env.fcn_tor(sim_rpm);

    %Angoli
    r       = sim_ang(1); %roll
    p       = sim_ang(2); %pitch
    y       = sim_ang(3); %yaw

    %Matrice rot w -> drn
    R = eul2rotm([y,p,r]);

    %Matrice rot vang 
    R_a = [ 1,   0,     -sin(p);
            0, cos(r), cos(p)*sin(r);
            0, -sin(r), cos(p)*cos(r)];

    %calcolo forze
    F_t = [0; 0; 1] * sum(thrust_arr);
    F_g = R \ (-[0; 0; 1] * env.drn_mass * env.g);
    F_a = R \ (-env.k0 * sim_vel .* abs(sim_vel));
    F_tot = F_t + F_g + F_a;

    %calcolo momenti
    %Momento forze
    M_x = env.arm_pos_y * thrust_arr;
    M_y = -env.arm_pos_x * thrust_arr;
    M_z = env.sign_z' * torque_arr;

    M_f = [M_x; M_y; M_z];

    % %Coppia giroscopica
    K_omega = sum(sim_wrot' * env.arm_sign_z');
    M_gyro = [
        -env.prp_in * sim_vang(2) * K_omega; ...
        env.prp_in * sim_vang(1) * K_omega; ...
        0
    ];
    % M_gyro = zeros(3,1);

    %Momento attrito
    M_a = -100 * env.torqueFric * sim_vang .* abs(sim_vang);

    % Calcolo momenti totali
    M_tot = M_gyro + M_f + M_a;

    %Integrazione
    sim_acc = F_tot/env.drn_mass - cross(sim_vang, sim_vel);
    sim_vel = sim_vel + R * sim_acc * env.dt;
    sim_pos = sim_pos + sim_vel * env.dt;

    %Blocco
    sim_pos(3) = max(sim_pos(3), 0);

    %Angoli
    sim_vacc = M_tot - cross(sim_vang, env.in_mat * sim_vang);
    sim_vacc = env.in_mat \ sim_vacc;
    sim_vang = sim_vang + sim_vacc * env.dt;
    sim_ang = sim_ang + R_a \ sim_vang * env.dt;

    % Ordina struct
    State.sim_ang = sim_ang;
    State.sim_vel = sim_vel;
    State.sim_pos = sim_pos;
    State.sim_vang = sim_vang;
    State.sim_time = sim_time;
    State.last_rpm = sim_rpm;
end
env.fcn_step = @stepState;

function out = randomPerm(values,N)
    combos = perms(values);
    combos = combos(:, 1:N);
    idx = randi(size(combos, 1));
    out = combos(idx, :)';
end
env.fcn_perm = @randomPerm;

function Action = predict(actor, obs) 
    int_action = getAction(actor, {obs});
    Action = int_action{1};
end
env.predict = @predict;

function Action = evalPid(State, env, com_arr)
        persistent integ;
        persistent lastAlt;  

        %First init
        if isempty(integ)
            integ = 0;  
            lastAlt = 0;
        end

        %Values from command array
        com_ang = com_arr(1:3);
        com_alt = com_arr(4);
        if not(com_alt == lastAlt)
            integ = 0;
        end

        %commanded angle
        err_ang = com_ang - State.sim_ang;

        %Comando
        Action = zeros(6,1);

        %%Angolo yaw
        kz = (err_ang(3) * 50 - 10 * State.sim_vang(3)) * env.arm_sign_z';
        kz = env.fcn_clamp(kz, -0.1, 0.1);
        Action = Action + kz;

        % %%Quota
        kz = (com_alt - State.sim_pos(3)) * 14 - 5 * State.sim_vel(3) + integ/10;
        kz = env.fcn_clamp(kz, 0.2, 0.7);
        Action = Action + kz * ones(6,1);

        err_z = com_alt - State.sim_pos(3);
        integ = integ + err_z * env.dt;
        integ = env.fcn_clamp(integ, -1, 1);
        lastAlt = com_alt;

        %%Posizione 
        %X
        kx = (8*err_ang(2) - 0.5*State.sim_vang(2)) * env.sign_x;
        kx = env.fcn_clamp(kx, -0.05, 0.05);
        Action = Action + kx;

        %Y
        ky = (8*err_ang(1) - 0.5*State.sim_vang(1)) * env.sign_y;
        ky = env.fcn_clamp(ky, -0.05, 0.05);
        Action = Action + ky;

        Action = env.fcn_clamp(Action, zeros(6,1), ones(6,1));
end
env.fcn_evalpid = @evalPid;
