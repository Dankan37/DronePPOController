%Modalità per VPS
try
    run("headlesscheck.mlx");
catch
    isHeadless = false;
end
disp("Headless server:" + num2str(isHeadless));
if isHeadless
    verbose = true;
    plotType = "none";
else
    verbose = false;
    plotType = "training-progress";
end