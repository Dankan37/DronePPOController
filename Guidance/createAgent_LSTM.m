numObs = 4;
numAct = 2;
numMap = 64;
numChan = 1;

obsInfo = [
    rlNumericSpec([numObs 1], ...
        Name="stateIn", ...
        Description="State vector (dx, dy, dz, AGL, heading error)");

    rlNumericSpec([numMap numMap numChan], ...
        Name="mapIn", ...
        Description="Local climb-cost patch (64x64x1)");
];

actInfo = rlNumericSpec([numAct 1], ...
    Name="action", ...
    Description="Local waypoint: (wx, wy, wz)");

actInfo.LowerLimit = [-1; 0];
actInfo.UpperLimit = [+1; +1];


%% === INPUT LAYERS ===
stateInput = featureInputLayer(numObs, Name="stateIn");

mapInput = imageInputLayer([numMap numMap numChan], ...
    Normalization="none", ...
    Name="mapIn");


%% === STATE BRANCH ===
stateBranch = [
    fullyConnectedLayer(64, Name="s_fc1")
    reluLayer(Name="s_relu1")
    fullyConnectedLayer(64, Name="s_fc2")
];


%% === PATCH BRANCH (CNN) ===
mapBranch = [
    convolution2dLayer(3,32,Padding="same",Name="m_conv1")
    layerNormalizationLayer(Name="m_bn1")
    reluLayer(Name="m_relu1")
    maxPooling2dLayer(2,Stride=2,Name="m_pool1")

    convolution2dLayer(3,64,Padding="same",Name="m_conv2")
    layerNormalizationLayer(Name="m_bn2")
    reluLayer(Name="m_relu2")
    maxPooling2dLayer(2,Stride=2,Name="m_pool2")

    convolution2dLayer(3,128,Padding="same",Name="m_conv3")
    layerNormalizationLayer(Name="m_bn3")
    reluLayer(Name="m_relu3")
    maxPooling2dLayer(2,Stride=2,Name="m_pool3")

    flattenLayer(Name="m_flat")
    fullyConnectedLayer(128,Name="m_fc1")
    reluLayer(Name="m_relu4")
];




%% === MERGE BRANCHES ===
merge = [
    concatenationLayer(1,2,"Name","concat")
    fullyConnectedLayer(128,"Name","fc_merge1")
    reluLayer("Name","relu_merge1")
    fullyConnectedLayer(64,"Name","fc_merge2")
    reluLayer("Name","relu_merge2")
    
];


%% === MEAN HEAD ===
meanHead = [
    fullyConnectedLayer(numAct,"Name","meanFC")
    tanhLayer("Name","tanhMean")
    scalingLayer("Scale", actInfo.UpperLimit, "Name","meanOut")  
];


%% === STD HEAD ===
stdHead = [
    fullyConnectedLayer(numAct,"Name","stdFC")
    softplusLayer
    scalingLayer("Scale",0.2, "Name","stdOut")
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
actorNet = connectLayers(actorNet, "m_relu4", "concat/in2");

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
c_stateInput = featureInputLayer(numObs, Name="c_stateIn");
c_mapInput   = imageInputLayer([numMap numMap numChan], Normalization="none", Name="c_mapIn");


%% === STATE BRANCH ===
c_stateBranch = [
    fullyConnectedLayer(64,"Name","cs_fc1")
    reluLayer("Name","cs_relu1")
    fullyConnectedLayer(64,"Name","cs_fc2")
];


%% === MAP BRANCH ===
c_mapBranch = [
    convolution2dLayer(3,32,Padding="same",Name="cm_conv1")
    layerNormalizationLayer(Name="cm_bn1")
    reluLayer(Name="cm_relu1")
    maxPooling2dLayer(2,Stride=2,Name="cm_pool1")

    convolution2dLayer(3,64,Padding="same",Name="cm_conv2")
    layerNormalizationLayer(Name="cm_bn2")
    reluLayer(Name="cm_relu2")
    maxPooling2dLayer(2,Stride=2,Name="cm_pool2")

    convolution2dLayer(3,128,Padding="same",Name="cm_conv3")
    layerNormalizationLayer(Name="cm_bn3")
    reluLayer(Name="cm_relu3")
    maxPooling2dLayer(2,Stride=2,Name="cm_pool3")

    flattenLayer(Name="cm_flat")
    fullyConnectedLayer(128,Name="cm_fc1")
    reluLayer(Name="cm_relu4")
];



%% === MERGE ===
c_merge = [
    concatenationLayer(1,2,"Name","c_concat")
    fullyConnectedLayer(128,"Name","c_fc1")
    reluLayer("Name","c_relu1")
    fullyConnectedLayer(64,"Name","c_fc2")
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
criticNet = connectLayers(criticNet, "cm_relu4", "c_concat/in2");

criticNet = dlnetwork(criticNet);


%% === RL WRAPPER ===
critic = rlValueFunction( ...
    criticNet, ...
    obsInfo, ...
    ObservationInputNames=["c_stateIn","c_mapIn"]);

%Agente & train
agentOpts_ext = rlPPOAgentOptions( ...
    ExperienceHorizon       = 256, ...
    ClipFactor              = 0.1, ...
    EntropyLossWeight       = 0.025, ...
    ActorOptimizerOptions   = rlOptimizerOptions(LearnRate=2e-5), ... %%4
    CriticOptimizerOptions  = rlOptimizerOptions(LearnRate=2e-5), ...
    MiniBatchSize           = 32, ...
    NumEpoch                = 4, ...
    DiscountFactor          = 0.995);

agent = rlPPOAgent(actor, critic, agentOpts_ext);