clear;
close all;
%%
%Modalità per VPS
try
    run("headlesscheck.mlx");
catch
    isHeadless = false;
end
disp("Headless server:" + num2str(isHeadless));
run("baseline_last.m");
%%
function [InitialObservation,State] = resetFunction(env)
    %Drone
    sim_pos = zeros(3,1);       %posizione
    sim_vel = zeros(3,1);       %velocità
    sim_ang = zeros(3,1);       %angoli
    sim_vang = zeros(3,1);      %vel angoli
    sim_int = zeros(3,1);

    %Curriculum training
    % 0 : pos fissa, tgt fisso
    % 1 : pos fissa, tgt random
    % 2 : pos random, tgt random
    % 3 : pos random, tgt random, angoli random
    % 4 : random waypoints
    sim_curr = env.sim_curr;

    %Initial state
    switch sim_curr
        case 0
            sim_pos = [0; 0; 10];
            sim_tgt = [5; 10; 10];
        case 1
            sim_pos = [0; 0; 10];
            sim_tgt = env.fcn_randomArrayBound([-10; -10; 10], [10; 10; 10]);
        case 2
            sim_pos = [0; 0; 10];
            sim_tgt = env.fcn_randomArrayBound([-10; -10; 5], [10; 10; 15]);
        case 3
            sim_pos = [0; 0; 10];
            sim_tgt = env.fcn_randomArrayBound([-15; -15; 5], [15; 15; 30]);
        case 4
            sim_pos = [0; 0; 10];
            sim_tgt = env.fcn_randomArrayBound([-15; -15; 5], [15; 15; 30]);
            sim_ang = env.fcn_randomArrayBound(-ones(3,1) * pi/5, ones(3,1) * pi/5);
        otherwise
            error("Caso non previsto")
    end

    %Compatibilità
    sim_tgt_pos = sim_tgt;

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
    Action = zeros(4,1);

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

function [NextObservation, Reward, IsDone, State] = stepFunction(Action, State, env)
    % EXT MODEL -> (attitude) -> INT MODEL

    %Output modello: 3 angoli eulero + variazione altitudine
    commanded_angles = Action(1:3);
    commanded_angles = env.fcn_clamp(commanded_angles, env.lowerlimit(1:3), env.upperlimit(1:3));
    delta_alt = State.sim_pos(3) * (1 + env.fcn_clamp(Action(4),-1,1));

    %Internal step
    for it = 1:8
        com_arr = [commanded_angles; delta_alt];

        int_act = env.fcn_evalpid(State, env, com_arr);

        NextState = env.fcn_step(State, int_act, env);
        State = NextState;
    end

    %Get States
    sim_vel     = State.sim_vel;
    sim_pos     = State.sim_pos;
    ang_nrm     = State.sim_ang / (pi/4);

    %Targets
    sim_tgt_pos     = State.sim_tgt_pos;

    %Err
    err_pos = (sim_tgt_pos - sim_pos) ./ 10;
    vel_nrm = sim_vel / 5;

    %Integrale
    sim_int = env.fcn_clamp(State.sim_int + err_pos * env.dt, -ones(3,1), ones(3,1));

    %EXTERNAL REWARD
    R_pos = -10 * norm(err_pos);
    R_vel = -0.1 * norm(vel_nrm);
    R_actSmooth  = -0.9 * sum((Action - State.last_act).^2);
    R_actNorm = -0.1 * sum(Action.^2);
    R_pen_vel = -5 * (State.sim_vel(3) < -3); %Penalità velocità verticale elevata

    Reward_ext = R_pos + R_vel + R_actSmooth + R_actNorm + R_pen_vel;

    Reward_ext = Reward_ext / 5;

    %Reward
    Reward = Reward_ext;

    %Exit
    IsDone = sim_pos(3) < 0.1 || any(abs(err_pos) > 5);
    if IsDone, Reward = Reward - 40; end

    %Win
    if all(abs(err_pos) < 0.02) && all(abs(vel_nrm) < 0.02)
        IsDone = true;
        Reward = Reward_ext + 40;
    end

    %Initial observation
    NextObservation = [err_pos; vel_nrm; ang_nrm; sim_int];

    %Save action
    State.last_act = Action;
    State.sim_int = sim_int;
end
%%
%SIMULAZIONE
% Imposta l'osservazione come l'output del reset
[InitialObservation,~] = resetFunction(env);
obsInfo = rlNumericSpec([size(InitialObservation,1) 1]);

%Modello esterno - Posizione
actInfo = rlNumericSpec([4 1]);

%Non utilizzati nel PPO
actInfo.LowerLimit = [-0.5;-0.5;-0.5; -1];
actInfo.UpperLimit = +[0.5; 0.5; 0.5; 1];
env.lowerlimit = actInfo.LowerLimit;
env.upperlimit = actInfo.UpperLimit;

%Ambiente
env.sim_curr = 0;
stepHandle = @(Action,State) stepFunction(Action, State, env);
ResetHandle = @() resetFunction(env);
tr_env = rlFunctionEnv(obsInfo,actInfo,stepHandle,ResetHandle);
%%
agentOpts_ext = rlPPOAgentOptions(...
    ExperienceHorizon       = 4000,...
    ClipFactor              = 0.1,...
    EntropyLossWeight       = 0.01,...
    ActorOptimizerOptions   = rlOptimizerOptions(LearnRate=3e-4),...
    CriticOptimizerOptions  = rlOptimizerOptions(LearnRate=6e-4),...
    MiniBatchSize           = 200,...
    NumEpoch                = 3,...
    SampleTime              = env.dt,...
    DiscountFactor          = 0.995);
%%
hlsz = 128;
initOpts = rlAgentInitializationOptions(NumHiddenUnit=hlsz);
agent = rlPPOAgent(obsInfo, actInfo, initOpts, agentOpts_ext);
%%
maxepisodes = 2500;
maxsteps = 500;

if isHeadless
    verbose = true;
    plotType = "none";
else
    verbose = false;
    plotType = "training-progress";
end

trainingOptions = rlTrainingOptions(...
    MaxEpisodes=maxepisodes,...
    MaxStepsPerEpisode=maxsteps,...
    StopOnError="on",...
    Verbose=verbose,...
    Plots=plotType,...
    ScoreAveragingWindowLength=30, ...
    StopTrainingCriteria="EvaluationStatistic", ...
    UseParallel = false, ...
    StopTrainingValue= 25000);
trainingOptions.ParallelizationOptions.Mode = "sync";

% agent evaluator
evl = rlEvaluator(EvaluationFrequency=30,NumEpisodes=5);

function dataToLog = myAgentLearnFinishedFcn(data)

    if mod(data.AgentLearnCount, 2) == 0
        dataToLog.ActorLoss  = data.ActorLoss;
        dataToLog.CriticLoss1 = max(data.CriticLoss);
        dataToLog.TD = mean(data.TDError);
    else
        dataToLog = [];
    end

end

function dataToLog = myEpisodeFinishedFcn(data)
    dataToLog.Lenght = length(data.Experience);
    dataToLog.Mean = mean(vertcat(data.Experience.Reward));
end

%Curriculum training automatico
for i = 1:4
    %Aggiorna addestramento
    env.sim_curr = i;

    %Reinizializza ambiene con env aggiornato
    stepHandle = @(Action,State) stepFunction(Action, State, env);
    ResetHandle = @() resetFunction(env);
    tr_env = rlFunctionEnv(obsInfo,actInfo,stepHandle,ResetHandle);

    %Plot
    if not(isHeadless)
        monitor = trainingProgressMonitor();
        logger = rlDataLogger(monitor);
        logger.AgentLearnFinishedFcn = @myAgentLearnFinishedFcn;
        logger.EpisodeFinishedFcn    = @myEpisodeFinishedFcn;
        trainingStats = train(agent,tr_env,trainingOptions,Evaluator=evl, Logger=logger);
    else
        trainingStats = train(agent,tr_env,trainingOptions,Evaluator=evl);
    end

    er = trainingStats.EpisodeReward;
    save("er_" + num2str(i), "trainingStats");

    %Salva ogni iterazione
    save("agent_" + num2str(i), "agent");
end

%Per SSH esci prima senza plot
if isHeadless
    return
end
%%
er = trainingStats.EpisodeReward;
plot(er);
%%
env.sim_curr = 3;

%Reinizializza ambiene con env aggiornato
stepHandle = @(Action,State) stepFunction(Action, State, env);
ResetHandle = @() resetFunction(env);
tr_env = rlFunctionEnv(obsInfo,actInfo,stepHandle,ResetHandle);

simOptions = rlSimulationOptions(MaxSteps=2500);
experience = sim(tr_env,agent,simOptions);

plot(experience.Reward)

%Posizione
figure
hold on
positionData = squeeze(experience.Observation.obs1.Data(1,1,:));
plot(positionData);

positionData = squeeze(experience.Observation.obs1.Data(2,1,:));
plot(positionData);

positionData = squeeze(experience.Observation.obs1.Data(3,1,:));
plot(positionData);
grid on
yline(0, 'r--', 'Target');
hold off
legend("X","Y","Z")
xlabel("N [step]"); ylabel("Errore normalizzato")
title("Errore posizione")

%Velocità
figure
hold on
positionData = squeeze(experience.Observation.obs1.Data(4,1,:));
plot(positionData);
positionData = squeeze(experience.Observation.obs1.Data(5,1,:));
plot(positionData);
positionData = squeeze(experience.Observation.obs1.Data(6,1,:));
plot(positionData);
yline(0, 'r--', '');
xlabel("N [step]")
ylabel("Velocità normalizzata")
grid on
hold off
legend("X","Y","Z")
title("Velocità")

%Angoli
figure
hold on
positionData = squeeze(experience.Observation.obs1.Data(7,1,:));
plot(positionData);
positionData = squeeze(experience.Observation.obs1.Data(8,1,:));
plot(positionData);
positionData = squeeze(experience.Observation.obs1.Data(9,1,:));
plot(positionData);
yline(0, 'r--', 'Target');
hold off
legend("X","Y","Z")
title("Angoli")


%Azioni
plot(experience.Action.act1)
legend()
