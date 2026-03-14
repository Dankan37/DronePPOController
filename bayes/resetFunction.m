function [InitialObservation,State] = resetFunction(env)
    %Drone
    sim_pos = zeros(3,1);       %posizione
    sim_vel = zeros(3,1);       %velocità
    sim_ang = zeros(3,1);       %angoli    
    sim_vang = zeros(3,1);      %vel angoli    
    sim_int = zeros(3,1);

    %Initial state
    sim_pos = [0; 0; 10];
    sim_tgt_pos = [5; 10; 10];
    
    %Motori
    sim_rpm = 3000 * ones(6,1);

    %Position error
    err_pos = (sim_tgt_pos - sim_pos) ./ 10;

    %Err
    vel_nrm = sim_vel / 5;
    ang_nrm = sim_ang / (pi/4);

    %Initial observation
    InitialObservation = [err_pos; vel_nrm; ang_nrm; sim_int];

    %Time
    sim_time = 0;

    %Last action
    Action = zeros(3,1);

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