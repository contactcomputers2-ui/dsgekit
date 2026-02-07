// New Keynesian 3-Equation Model
// IS curve, Phillips curve, Taylor rule with interest rate smoothing

var x pi i;
varexo e_d e_s e_m;

parameters sigma beta kappa phi_pi phi_x rho_i;

sigma  = 1.0;
beta   = 0.99;
kappa  = 0.1;
phi_pi = 1.5;
phi_x  = 0.5;
rho_i  = 0.8;

model;
    [is_curve] x = x(+1) - sigma * (i - pi(+1)) + e_d;
    [phillips] pi = beta * pi(+1) + kappa * x + e_s;
    [taylor]   i = rho_i * i(-1) + (1 - rho_i) * (phi_pi * pi + phi_x * x) + e_m;
end;

initval;
    x  = 0;
    pi = 0;
    i  = 0;
end;

shocks;
    var e_d; stderr 0.01;
    var e_s; stderr 0.01;
    var e_m; stderr 0.01;
end;
