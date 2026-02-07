// AR(1) transition with deterministic shocks + boundary blocks
// Tests: DYN-D02 integration path reproducibility

var y;
varexo e;
parameters rho;

rho = 0.9;

model;
    y = rho * y(-1) + e;
end;

initval;
    y = 0.0;
end;

histval;
    y(0) = 0.2;
end;

endval;
    y = 0.0;
end;

shocks;
    var e; stderr 0.0;
    var e;
        periods 1:3;
        values 0.05;
end;
