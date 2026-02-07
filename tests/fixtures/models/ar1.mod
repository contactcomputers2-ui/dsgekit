// Simple AR(1) Model
// For testing dsgekit parser

var y;
varexo e;

parameters rho;

rho = 0.9;

model;
    [ar1] y = rho * y(-1) + e;
end;

initval;
    y = 0;
end;

shocks;
    var e; stderr 0.01;
end;
