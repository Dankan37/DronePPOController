clear;
close all;
clc;

global save_pos;
save_pos = false;

run("sshutils.m");
run("scan_utils.m");

%Mappa ambiente
env.F = F;
env.XY_bound = XY_bound;
env.scan_params.max_distance = 300;
env.scan_params.step_size   = 2;
env.scan_ang  = 45;
env.scan_step = 1;

% Resolution and map size
env.resolution = 2;
env.patchSize  = 64;

% Compute dynamic map dimensions
W = round((XY_bound.xmax - XY_bound.xmin) / env.resolution) + 1;
H = round((XY_bound.ymax - XY_bound.ymin) / env.resolution) + 1;

% Initialize persistent memory
env.heightmap    = nan(H, W);
env.height_known = false(H, W);

%Agent
run("createAgent_LSTM.m");
load("agent_big_300_2.mat")
agent.AgentOptions = agentOpts_ext;

if ~isHeadless
    start_pos  = [0;   25;   0] + [randi(m_size, 1, 1); 0; 0];
    target_pos = [0;   m_size - 25;   0] + [randi(m_size, 1, 1); 0; 0];

    start_pos(3)  = F(start_pos(1), start_pos(2)) + 5;
    target_pos(3) = F(target_pos(1), target_pos(2)) + 5;

    % Plot the heightmap
    figure; hold on;
    surf(Xgrid, Ygrid, Zgrid, 'EdgeColor','none','FaceAlpha',0.85);
    shading interp; colormap parula;

    % Plot start and target positions
    plot3(start_pos(1), start_pos(2), start_pos(3), 'ro', 'MarkerSize', 10, 'DisplayName', 'Start Position');
    plot3(target_pos(1), target_pos(2), target_pos(3), 'go', 'MarkerSize', 10, 'DisplayName', 'Target Position');

    title(sprintf('Vista 3d'));
    view(45,35); grid on; axis tight;

    % Add legend
    legend show;
    hold off;
end
env.size = m_size; env.save = false;


function [InitialObservation, State] = ResetEnv(env)

    % unpack
    F = env.F;
    B = env.XY_bound;
    m_size = env.size;

    % start and target
    start_pos  = [0;   25;   0] + [randi(m_size, 1, 1); 0; 0];
    target_pos = [0;   m_size - 25;   0] + [randi(m_size, 1, 1); 0; 0];

    global save_pos
    if save_pos
        save("pos_sim","start_pos","target_pos");
    end

    % fixed altitude
    desiredAGL = 5;
    start_pos(3)  = F(start_pos(1),  start_pos(2))  + desiredAGL;
    target_pos(3) = F(target_pos(1), target_pos(2)) + desiredAGL;

    % initial yaw
    % Convention: yaw = 0 → +Y (north), +90 → +X (east)
    dx = target_pos(1) - start_pos(1);
    dy = target_pos(2) - start_pos(2);
    start_yaw = atan2d(dx, dy);

    % reset map memory
    env.heightmap(:)    = nan;
    env.height_known(:) = false;

    % initial radar scans
    % Scan from drone position
    env = scanRadar(start_pos, start_yaw, env);

    % Additional forward scan 
    fwd = [sind(start_yaw); cosd(start_yaw)];
    scan_pos = start_pos;
    scan_pos(1:2) = scan_pos(1:2) + 10 * fwd;

    env = scanRadar(scan_pos, start_yaw, env);

    % state vector
    scale = env.size;

    dx_norm = (target_pos(1) - start_pos(1)) / scale;
    dy_norm = (target_pos(2) - start_pos(2)) / scale;
    dz_norm = (target_pos(3) - start_pos(3)) / scale;
    heading_err = 0;   % aligned at reset

    stateVec = [dx_norm; dy_norm; start_pos(3)/20; heading_err];

    % forward patch
    [rawPatch, knownMask] = getLocalPatch(env, scan_pos, start_yaw, env.resolution);
    climbMap = makeClimbMap(rawPatch, start_pos(3), knownMask);
    mapIn = fliplr(climbMap);

    % state struct
    State.pos         = start_pos;
    State.yaw         = start_yaw;
    State.tgt         = target_pos;
    State.env         = env;
    State.scale_factor = scale;
    State.last_dist   = norm(start_pos(1:2) - target_pos(1:2));
    State.prevAction  = [0; 0];

    % output
    InitialObservation = {stateVec, mapIn};

end


function [nextObv, Reward, IsDone, State] = stepFcn(Action, State)

    % unpack
    env = State.env;
    F   = env.F;

    pos = State.pos;     % [x; y; z]
    yaw = State.yaw;     % degrees
    tgt = State.tgt;

    scale = State.scale_factor;

    % action as local waypoint
    Action = max(min(Action, env.upperlimit), env.lowerlimit);

    wx = 0.3*Action(1);   % right / left
    wy = Action(2);   % forward

    % local waypoint to world
    lookahead = 15;               % meters 

    % Local → world rotation
    % Local: X=right, Y=forward
    fwd  = [sind(yaw); cosd(yaw)];
    left = [-cosd(yaw); sind(yaw)];
    
    waypointXY = pos(1:2) ...
               + wy * lookahead * fwd ...
               + wx * lookahead * left;

    % move toward waypoint
    maxStep = 5.0;   % meters per step

    dir = waypointXY - pos(1:2);
    distXY = norm(dir);

    if distXY > 1e-6
        dir = dir / distXY;
    else
        dir = [0;0];
    end

    stepXY = dir * min(distXY, maxStep);
    new_xy = pos(1:2) + stepXY;

    % yaw update
    newYaw = atan2d(dir(1), dir(2));   % atan2(dx, dy)
    newYaw = wrapTo180(newYaw);


    % altitude handling
    desiredAGL = 5;
    groundZ = F(new_xy(1), new_xy(2));
    new_z   = groundZ + desiredAGL;

    new_pos = [new_xy; new_z];

    % target metrics
    dist        = norm(new_xy - tgt(1:2));
    prevDist    = State.last_dist;

    dx_t = tgt(1) - new_xy(1);
    dy_t = tgt(2) - new_xy(2);
    tgtBearing = atan2d(dx_t, dy_t);

    headingErr  = wrapTo180(tgtBearing - newYaw) / 90;

    dx = (tgt(1) - new_xy(1)) / scale;
    dy = (tgt(2) - new_xy(2)) / scale;
    dz = (tgt(3) - new_z)    / scale;

    % update map
    env = scanRadar(new_pos, newYaw, env);
    State.env = env;

    % Forward-looking patch
    fwd = [cosd(newYaw); sind(newYaw)];
    scan_pos = new_pos;
    scan_pos(1:2) = scan_pos(1:2) + 5 * fwd;

    [rawPatch, knownMask] = getLocalPatch(env, scan_pos, newYaw, env.resolution);
    climbMap = makeClimbMap(rawPatch, new_pos(3), knownMask);
    mapIn = fliplr(climbMap);

    % observation
    stateVec = [dx; dy; new_pos(3)/20; headingErr];

    % reward
    deltaDist = prevDist - dist;
    R_progress = 6.0 * deltaDist / scale;

    %Alt penality
    R_z = -0.4 * new_pos(3);
    R_heading = 0.5 * cosd(90 * headingErr);
    R_goal = -1.2 * (dist / env.size);

    % assume climbMap defined such that positive = free
    minClearance = min(min(climbMap(30:34,30:40)));  % worst point in patch
    R_clear = -0.68 * exp(-0.5 * minClearance);

    Reward = R_goal + R_progress + R_clear - 0.33 * abs(wx);

    % termination
    IsDone = false;

    if dist < 15
        Reward = Reward + 50;
        IsDone = true;
    end

    MAX_DZ_STEP = 30;
    if abs(new_pos(3) - pos(3)) >= MAX_DZ_STEP
        Reward = Reward - 20;
        IsDone = true;
    end

    if new_xy(1) < env.XY_bound.xmin || new_xy(1) > env.XY_bound.xmax || ...
       new_xy(2) < env.XY_bound.ymin || new_xy(2) > env.XY_bound.ymax
        Reward = Reward - 20;
        IsDone = true;
    end

    % update state
    State.pos        = new_pos;
    State.yaw        = newYaw;
    State.last_dist  = dist;
    State.prevAction = Action;

    % output
    nextObv = {stateVec, mapIn};

end


%Ambiente
env.lowerlimit = actInfo.LowerLimit;
env.upperlimit = actInfo.UpperLimit;

resetHandle = @()ResetEnv(env);
stepHandle = @(Action,State) stepFcn(Action, State);
tr_env = rlFunctionEnv(obsInfo,actInfo,stepHandle,resetHandle);

testAgent(agent, env, isHeadless, resetHandle, stepHandle);

maxepisodes = 55000;
maxsteps = 140;


trainingOptions = rlTrainingOptions(...
    MaxEpisodes=maxepisodes,...
    MaxStepsPerEpisode=maxsteps,...
    StopOnError="on",...
    Verbose=verbose,...
    Plots=plotType,...
    ScoreAveragingWindowLength=30, ...
    StopTrainingCriteria="AverageReward", ...
    SaveAgentCriteria="EpisodeFrequency",...
    SaveAgentValue=50,... 
    UseParallel = false, ...
    StopTrainingValue= 15000); 

% agent evaluator
save_pos = false;
evl = rlEvaluator(EvaluationFrequency=100,NumEpisodes=5);
trainingStats = train(agent,tr_env,trainingOptions,Evaluator=evl);

if isHeadless
    return
end

save_pos = true;
simOptions = rlSimulationOptions(MaxSteps=maxsteps);
experience = sim(tr_env,agent,simOptions);


experience.Reward.Data(end) = experience.Reward.Data(end) / 50;
plot(experience.Reward)

%Posizione
figure
hold on
positionData = squeeze(experience.Observation.stateIn.Data(1,1,:));
plot(positionData);

positionData = squeeze(experience.Observation.stateIn.Data(2,1,:));
plot(positionData);

positionData = squeeze(experience.Observation.stateIn.Data(3,1,:));
plot(positionData);

yline(0, 'r--', 'Target'); 
hold off
legend("X","Y","Z","POS")
title("Posizioni")

positionData = squeeze(experience.Observation.stateIn.Data(4,1,:));
plot(positionData);


%Azioni
plot(experience.Action.action)
legend("BRG","SPT")
for i = 1:size(experience.Observation.mapIn.Data,3)
    if mod(i,10) == 1
        pm = experience.Observation.mapIn.Data(:,:,i);
        figure;
        imagesc(pm, [-1 1]);  % proper grayscale range
        colormap(gray(256));
        colorbar;
        axis equal tight;
        title(sprintf('Patch Map t=%d', i));
    end
end
load("pos_sim.mat")
pos_arr = [];
for i = 1:3
    pos_i = squeeze(experience.Observation.stateIn.Data(i,1,:));  
    pos_arr(i,:) = target_pos(i) - pos_i * env.size;
end

pos_arr(3,:) = F(pos_arr(1,:),pos_arr(2,:)) + 5;


run("scan_utils.m")
figure; hold on;
surf(Xgrid, Ygrid, Zgrid.', 'EdgeColor','none')
shading interp; colormap parula;
for i = 1:length(pos_arr)
    pos = pos_arr(:,i);
    plot3(pos(1),pos(2),pos(3),'ro', 'MarkerSize', 3 );
end
plot3(target_pos(1),target_pos(2),target_pos(3),'ro', 'MarkerSize', 12 );
hold off
view(45,35); grid on; axis tight;
