// New Keynesian model with estimation blocks
// Tests: varobs, estimated_params, steady_state_model

var x pi i;
varexo e_d e_s e_m;
parameters sigma beta kappa phi_pi phi_x rho_i;

sigma = 1.0;
beta = 0.99;
kappa = 0.1;
phi_pi = 1.5;
phi_x = 0.5;
rho_i = 0.8;

model;
    [is_curve] x = x(+1) - sigma * (i - pi(+1)) + e_d;
    [phillips] pi = beta * pi(+1) + kappa * x + e_s;
    [taylor] i = rho_i * i(-1) + (1 - rho_i) * (phi_pi * pi + phi_x * x) + e_m;
end;

varobs x pi i;

steady_state_model;
    x = 0;
    pi = 0;
    i = 0;
end;

estimated_params;
    phi_pi, 1.5, 1.0, 3.0, normal_pdf, 1.5, 0.25;
    phi_x, 0.5, 0.0, 2.0, gamma_pdf, 0.5, 0.25;
    stderr e_d, 0.01, 0.001, 0.1;
    stderr e_s, 0.01, 0.001, 0.1;
end;

shocks;
    var e_d; stderr 0.01;
    var e_s; stderr 0.01;
    var e_m; stderr 0.01;
end;
