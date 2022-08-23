%% Function Simulation
close all
clear 
clc

N = 1000;
T = 60;
N_s = 1000;
N_p = 9;  
min_p = 0.001;
max_p = 0.003;

p = linspace(min_p, max_p, N_p)';
c = zeros(N_p, 1);
for i = 1:N_p
    I = 1*ones(N_s, 1);
    S = N*ones(N_s, 1) - I;
    for j = 1:T
        I = binornd(S, 1 - (1-p(i)).^I);
        S = S - I;
    end
    c(i) = (0.003/p(i)).^9-1 + N - mean(S);
end

%% Normalizing

p = (p - min_p)/(max_p - min_p);
p = [p, ones(N_p, 1)];
min_c = min(c);
max_c = max(c);
c = (c - min_c)/(max_c - min_c);

%% Neural Network

H = 3;
w = unifrnd(-0.5, 0.5, 1+1, H);
k = unifrnd(-0.5, 0.5, H+1, 1);
tol = 1.0e-7;
SSE_old = 1.0e+11;
SSE_new = 1.0e+10;

m = 1;
while (SSE_old - SSE_new)/SSE_old > tol || m < 10000
    SSE_old = SSE_new;
    mu = 5000/(100000 + m);
    v_star = p*w;
    v = [1./(1 + exp(-v_star)), ones(N_p, 1)];
    o = v*k;
    w = w + 2*mu*p'*((c-o)*k(1:H)'.*v(:, 1:H).*(1-v(:, 1:H)));
    k = k + 2*mu*v'*(c-o);
    SSE_new = (c-o)'*(c-o);
    m = m+1;
end

%% Finding optimal p

N_points = 101;
x = [linspace(0, 1, N_points)', ones(N_points, 1)];
v_star = x*w;
v = [1./(1 + exp(-v_star)), ones(N_points, 1)];
y = v*k;
[c_star, index] = min(y);

%% Plot

fx = @(x) x*(max_p - min_p) + min_p;
fy = @(y) y*(max_c - min_c) + min_c;
figure('Name','Neural Network','NumberTitle','off')
hold on
plot(fx(p(:, 1)), fy(c), 'ro')
plot(fx(x(:, 1)), fy(y), 'b-')
plot(fx(x(index, 1)), fy(c_star), 'g*')
hold off
xlabel('p')
ylabel('c')

