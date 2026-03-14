function climbMap = makeClimbMap(patch, droneAlt, knownMask)

    climbMap = zeros(size(patch));   % UNKNOWN = NEUTRAL

    valid = knownMask & ~isnan(patch);

    if any(valid(:))
        clearance = droneAlt - patch(valid);
        normClear = clearance / 20;          % ±20 m range
        normClear = max(-1, min(1, normClear));

        climbMap(valid) = normClear;
    end
end
