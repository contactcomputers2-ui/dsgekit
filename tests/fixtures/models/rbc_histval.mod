// RBC model with histval and endval blocks
// Tests: histval, endval

var y c k a;
varexo e;
parameters beta alpha rho delta;

beta = 0.99;
alpha = 0.33;
rho = 0.95;
delta = 0.025;

model;
    [euler] 1/c = beta * (1/c(+1)) * (alpha * a(+1) * k^(alpha-1) + 1 - delta);
    [production] y = a * k(-1)^alpha;
    [resource] y = c + k - (1-delta)*k(-1);
    [technology] log(a) = rho * log(a(-1)) + e;
end;

initval;
    a = 1;
    k = 10;
    y = 1;
    c = 0.8;
end;

endval;
    a = 1;
    k = 12;
    y = 1.2;
    c = 0.9;
end;

histval;
    k(0) = 9.5;
    a(0) = 1.0;
    a(-1) = 0.98;
end;

shocks;
    var e; stderr 0.01;
end;
