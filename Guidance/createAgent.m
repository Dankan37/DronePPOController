obsInfo = [
    rlNumericSpec([5 1], ...
        Name="stateIn", ...
        Description="State vector (dx, dy, dz, AGL, heading error)");

    rlNumericSpec([64 64 1], ...
        Name="mapIn", ...
        Description="Local climb-cost patch (64x64x1)");
];

actInfo = rlNumericSpec([2 1], ...
    Name="action", ...
    Description="Local waypoint: (wx, wy, wz)");

actInfo.LowerLimit = [-5; 0];
actInfo.UpperLimit = [+5; +1];


%% === INPUT LAYERS ===
stateInput = featureInputLayer(5, Name="stateIn");

mapInput = imageInputLayer([64 64 1], ...
    Normalization="none", ...
    Name="mapIn");


%% === STATE BRANCH ===
stateBranch = [
    fullyConnectedLayer(64, Name="s_fc1")
    reluLayer(Name="s_relu1")
    fullyConnectedLayer(32, Name="s_fc2")
];


%% === PATCH BRANCH (CNN) ===
mapBranch = [
    convolution2dLayer(3,16,Padding="same",Name="m_conv1")
    batchNormalizationLayer
    reluLayer(Name="m_relu1")
    maxPooling2dLayer(2,Stride=2,Name="m_pool1")

    convolution2dLayer(3,32,Padding="same",Name="m_conv2")
    batchNormalizationLayer
    reluLayer(Name="m_relu2")
    maxPooling2dLayer(2,Stride=2,Name="m_pool2")

    flattenLayer(Name="m_flat")
    fullyConnectedLayer(64,Name="m_fc1")
    reluLayer(Name="m_relu3")
];


%% === MERGE BRANCHES ===
merge = [
    concatenationLayer(1,2,"Name","concat")
    fullyConnectedLayer(64,"Name","fc_merge1")
    reluLayer("Name","relu_merge1")
    fullyConnectedLayer(32,"Name","fc_merge2")
    reluLayer("Name","relu_merge2")
    
];


%% === MEAN HEAD ===
meanHead = [
    fullyConnectedLayer(2,"Name","meanFC")
    tanhLayer("Name","tanhMean")
    scalingLayer("Scale", actInfo.UpperLimit, "Name","meanOut")  
];


%% === STD HEAD ===
stdHead = [
    fullyConnectedLayer(2,"Name","stdFC")
    softplusLayer("Name","stdOut")
];


%% === BUILD ACTOR DLNETWORK ===

actorNet = layerGraph();

actorNet = addLayers(actorNet, stateInput);
actorNet = addLayers(actorNet, stateBranch);
actorNet = addLayers(actorNet, mapInput);
actorNet = addLayers(actorNet, mapBranch);
actorNet = addLayers(actorNet, merge);
actorNet = addLayers(actorNet, meanHead);
actorNet = addLayers(actorNet, stdHead);

% connections
actorNet = connectLayers(actorNet, "stateIn", "s_fc1");
actorNet = connectLayers(actorNet, "mapIn", "m_conv1");

actorNet = connectLayers(actorNet, "s_fc2", "concat/in1");
actorNet = connectLayers(actorNet, "m_relu3", "concat/in2");

actorNet = connectLayers(actorNet, "relu_merge2", "meanFC");
actorNet = connectLayers(actorNet, "relu_merge2", "stdFC");


% convert to dlnetwork
actorNet = dlnetwork(actorNet);

%% === RL WRAPPER ===
actor = rlContinuousGaussianActor( ...
    actorNet, ...
    obsInfo, ...
    actInfo, ...
    ObservationInputNames=["stateIn","mapIn"], ...
    ActionMeanOutputNames="meanOut", ...
    ActionStandardDeviationOutputNames="stdOut");

%% === CRITIC INPUTS ===
c_stateInput = featureInputLayer(5, Name="c_stateIn");
c_mapInput   = imageInputLayer([64 64 1], Normalization="none", Name="c_mapIn");


%% === STATE BRANCH ===
c_stateBranch = [
    fullyConnectedLayer(64,"Name","cs_fc1")
    reluLayer("Name","cs_relu1")
    fullyConnectedLayer(32,"Name","cs_fc2")
];


%% === MAP BRANCH ===
c_mapBranch = [
    convolution2dLayer(3,16,Padding="same",Name="cm_conv1")
    batchNormalizationLayer
    reluLayer(Name="cm_relu1")
    maxPooling2dLayer(2,Stride=2,Name="cm_pool1")

    convolution2dLayer(3,32,Padding="same",Name="cm_conv2")
    batchNormalizationLayer
    reluLayer(Name="cm_relu2")
    maxPooling2dLayer(2,Stride=2,Name="cm_pool2")

    flattenLayer(Name="cm_flat")
    fullyConnectedLayer(64,"Name","cm_fc1")
    reluLayer("Name","cm_relu3")
];


%% === MERGE ===
c_merge = [
    concatenationLayer(1,2,"Name","c_concat")
    fullyConnectedLayer(64,"Name","c_fc1")
    reluLayer("Name","c_relu1")
    fullyConnectedLayer(32,"Name","c_fc2")
    reluLayer("Name","c_relu2")
    fullyConnectedLayer(1,"Name","value")   % <-- single output!
];


%% === BUILD CRITIC NETWORK ===
criticNet = layerGraph();

criticNet = addLayers(criticNet, c_stateInput);
criticNet = addLayers(criticNet, c_stateBranch);
criticNet = addLayers(criticNet, c_mapInput);
criticNet = addLayers(criticNet, c_mapBranch);
criticNet = addLayers(criticNet, c_merge);

criticNet = connectLayers(criticNet, "c_stateIn", "cs_fc1");
criticNet = connectLayers(criticNet, "c_mapIn",   "cm_conv1");

criticNet = connectLayers(criticNet, "cs_fc2", "c_concat/in1");
criticNet = connectLayers(criticNet, "cm_relu3", "c_concat/in2");

criticNet = dlnetwork(criticNet);


%% === RL WRAPPER ===
critic = rlValueFunction( ...
    criticNet, ...
    obsInfo, ...
    ObservationInputNames=["c_stateIn","c_mapIn"]);

%Agente & train
agentOpts_ext = rlPPOAgentOptions( ...
    ExperienceHorizon       = 512, ...
    ClipFactor              = 0.2, ...
    EntropyLossWeight       = 0.025, ...
    ActorOptimizerOptions   = rlOptimizerOptions(LearnRate=5e-5), ...
    CriticOptimizerOptions  = rlOptimizerOptions(LearnRate=1e-4), ...
    MiniBatchSize           = 64, ...
    NumEpoch                = 4, ...
    DiscountFactor          = 0.995);

agent = rlPPOAgent(actor, critic, agentOpts_ext);