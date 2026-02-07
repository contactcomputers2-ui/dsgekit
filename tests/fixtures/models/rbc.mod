// Simple RBC Model
// For testing dsgekit parser

var y c k a;
varexo e;

parameters beta alpha rho delta;

beta = 0.99;
alpha = 0.33;
rho = 0.95;
delta = 0.025;

model;
    // Euler equation
    [euler] 1/c = beta * (1/c(+1)) * (alpha * a(+1) * k^(alpha-1) + 1 - delta);

    // Production function
    [production] y = a * k(-1)^alpha;

    // Resource constraint
    [resource] y = c + k - (1-delta)*k(-1);

    // Technology shock
    [technology] log(a) = rho * log(a(-1)) + e;
end;

initval;
    a = 1;
    k = 28.3484190610;
    y = 3.0153277085;
    c = 2.3066172320;
end;

shocks;
    var e; stderr 0.01;
end;
