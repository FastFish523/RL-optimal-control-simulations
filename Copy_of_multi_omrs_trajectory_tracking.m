clc;
clear;
close all;
% Simulation parameters
t_span = linspace(0, 10, 2000);     % Time span for simulation. 
timeScale=200;
N = 4;      % Number of vehicles

% Control and system parameters
Ct = 0.3;
R = 0.0635;
L = 0.14;
m = 5;
mI = 0.1125;
[Lap, B_comm] = communication_topo();

% Designed parameters
K_eta = 20;
K_q = 20;
Gamma = 2;
R_u = diag([1,1,1]);
alpha_w = 500;
Q_e = diag([1,1,1]);

num_weights = 6;
num_states = 6;
num_inputs = 3;
% Design positive definite matrix
q_bar = (Lap + B_comm) \ ones(N, 1);
P = diag(1 ./ q_bar);

% Define system matrices
M = diag([m, m, mI]);
F_theta = @(theta) [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];
C = @(omega) [0, m*omega, 0; -m*omega, 0, 0; 0, 0, 0];
B = [1/2, 1/2, -1; sqrt(3)/2, -sqrt(3)/2, 0; L, L, L];
g1 = @(state) F_theta(state(3));
f2 = @(state) -M \ C(state(6)) * state(4:6);
g2 = M \ (Ct / R * B);
% Caculus functions
dot_inv_g1=@(state) [-sin(state(3))*state(6), cos(state(3))*state(6), 0; -cos(state(3))*state(6), -sin(state(3))*state(6), 0; 0, 0, 0];

% % Reference trajectory
% eta_r = @(t) [t; sin(t); 0];
% eta_r_dot = @(t) [1;cos(t); 0];
% eta_r_ddot = @(t) [0;-sin(t); 0];
eta_r = @(t) [5*t; 5*sin(t); 1];
eta_r_dot = @(t) [5;5*cos(t); 0];
eta_r_ddot = @(t) [0;-5*sin(t); 0];

% Linear paramized form
bar_fr=@(t) [t,sin(t),0;1,1,1];
dot_bar_fr=@(t) [1,cos(t),0;0,0,0];
ddot_bar_fr=@(t) [0,sin(t),0;0,0,0];

% Initial para
Hatbar_wr = zeros(2,length(t_span),N);
State=zeros(6,length(t_span),N);
U=zeros(3,length(t_span),N);
actor_update = zeros(num_inputs,1);
W1 = zeros(num_weights,length(t_span),N);
Alpha1 = zeros(3,length(t_span),N);
% Set initial values
% hatbar_wr_0 = [0,0]';
% state_0 = [0.5, 0.6, 0.7, 1, -1, 2]';      % Initial state for all vehicles
% Hatbar_wr(:,1,1) = hatbar_wr_0;
% Hatbar_wr(:,1,2) = hatbar_wr_0;
% Hatbar_wr(:,1,3) = hatbar_wr_0;
% Hatbar_wr(:,1,4) = hatbar_wr_0;
% State(:,1,1) = state_0;
% State(:,1,2) = state_0;
% State(:,1,3) = state_0;
% State(:,1,4) = state_0;
% State_multiple=ones(6,4) .* state_0(1:6);       % 3*4
State_multiple=ones(6,4);
for i = 1 : N
%     Hatbar_wr(:,1,i) =[5;0];
    Hatbar_wr(:,1,i) =rand(2,1);
%     State(:,1,i) = zeros(6,1);
    State(:,1,i) = rand(6,1);
    State_multiple(:,i)=State(:,1,i);       % 3*4
end
W1(:,1,:)=1;

% eta_r = @(t) [5 * sin(t); 5 * sin(t) * cos(t); 1 * t];
% eta_r_dot = @(t) [5 * cos(t); 5 * cos(2*t) - 5 * sin(t) * sin(t); 1];
% eta_r_ddot = @(t) [-5 * sin(t); -10*sin(2*t) - 5 * cos(2*t) * sin(t); 0];

% Reference values for plotting
eta_r_vals = zeros(3, length(t_span));
for i = 1:length(t_span)
    eta_r_vals(:, i) = eta_r(t_span(i));
end
% Prepare figure for real-time plotting
figure;
hold on;
colors = ['r', 'g', 'b', 'm']; % Different colors for different vehicles
plot_handles = gobjects(1, N);
for i = 1:N
    plot_handles(i) = plot(State(1, 1, i), State(2, 1, i), [colors(i), '-o'], 'DisplayName', ['OMR ', num2str(i)]);
end
plot(eta_r_vals(1, :), eta_r_vals(2, :), 'k--', 'DisplayName', 'Reference Trajectory');
legend;
xlabel('x');
ylabel('y');
title('Trajectories of Unmanned OMRs');
hold off;

% Prepare figure for real-time W1 plotting
figure;
hold on;
w1_colors = ['r', 'g', 'b', 'm', 'c', 'k']; % Different colors for different weights
w1_plot_handles = gobjects(num_weights,1);
for i = 1:num_weights
    w1_plot_handles(i) = plot(t_span(1), W1(i, 1, 1), [w1_colors(i), '-'], 'DisplayName', ['W1', num2str(i),' OMR1 ']);
end
legend;
xlabel('Time (s)');
ylabel('W1 Values');
title('Evolution of W1 Weights');
hold off;
for episode = 1:1
    State_multiple=ones(6,4);
    for i = 1 : N
        %     Hatbar_wr(:,1,i) =[5;0];
        Hatbar_wr(:,1,i) =rand(2,1);
        %     State(:,1,i) = zeros(6,1);
        State(:,1,i) = rand(6,1);
        State_multiple(:,i)=State(:,1,i);       % 3*4
    end
    for steps = 2:length(t_span)
        for i=1:N
            t_step = t_span(steps-1:steps);

            % Control law definition
            control_law = @(t, state) single_vehicle_control_law(t, state,dot_inv_g1, State_multiple, i, g1, g2, f2, eta_r, eta_r_dot, eta_r_ddot, K_eta, K_q, Gamma, Lap, B_comm, P, N, bar_fr, dot_bar_fr, ddot_bar_fr,Hatbar_wr(:,steps - 1,i));

            % Optimal control law definition
            optimal_control = @(t, state) single_vehicle_optimal_control(t, state,dot_inv_g1, State_multiple, i, g1, g2, f2, eta_r, eta_r_dot, eta_r_ddot, K_eta, K_q, Gamma, Lap, B_comm, P, N, bar_fr, dot_bar_fr, ddot_bar_fr,Hatbar_wr(:,steps - 1,i),W1(:,steps - 1,i),R_u,num_weights,alpha_w,Q_e,timeScale);

            % Define the dynamics for multiple vehicles
            dynamics = @(t, state) single_vehicle_dynamics(t, state, control_law, optimal_control, g1,f2,g2);

            % Solve ODE parameter update
            [t, states] = ode45(dynamics, t_step, State(:,steps - 1,i));
            State(:,steps,i) = states(end,:);
            State_multiple(:,i)=states(end,:);
            optimal_results = single_vehicle_optimal_control(t_step(1),State(:,steps-1,i),dot_inv_g1, State_multiple, i, g1, g2, f2, eta_r, eta_r_dot, eta_r_ddot, K_eta, K_q, Gamma, Lap, B_comm, P, N, bar_fr, dot_bar_fr, ddot_bar_fr,Hatbar_wr(:,steps - 1,i),W1(:,steps - 1,i),R_u,num_weights,alpha_w,Q_e,timeScale);
            W1(:,steps,i) = optimal_results(4:4+num_weights-1);
            %         Alpha1(:,steps,i) = optimal_results(1:3);

            U(:, steps, i) = control_law(t_span(steps), State(:,steps,i));

            % Define parameter update
            para_update = @(t,hatbar_wr) leader_trajectory_estimate_law(t, i, State_multiple, Gamma, eta_r, bar_fr, Lap, B_comm);

            % Solve ODE parameter update law
            [~,hbar_wrs] = ode45(para_update,t_step,Hatbar_wr(:,steps - 1,i));
            %         Hatbar_wr(:,steps,i) = [5;0];
            Hatbar_wr(:,steps,i) = hbar_wrs(end,:);

            % Update plot data
            set(plot_handles(i), 'XData', State(1, 1:steps, i), 'YData', State(2, 1:steps, i));
            if i == 1  % Only update W1 plot for OMR1
                for j = 1:num_weights
                    set(w1_plot_handles(j), 'XData', t_span(1:steps), 'YData', W1(j, 1:steps, 1));
                end
            end
            W1(:,1,i)=W1(:,steps,i);
        end
        drawnow; % Update the plot
    end
    % save('all_Alpha1.mat','Alpha1');
end

% Plot results
figure;
for i = 1:N
    subplot(N, 1, i);
    plot(t_span, Hatbar_wr(1,:,i), 'r', t_span, 5*ones(size(t_span)), 'b--');
    hold on;
    plot(t_span, Hatbar_wr(2,:,i), 'g', t_span, zeros(size(t_span)), 'm--');
    legend('wr-e', 'wr-t', 'cr-e', 'cr-t');
    title(['OMR ', num2str(i), ' Reference Trajectory Estimate']);
    xlabel('Time (s)');
    ylabel('Values');
    hold off;
end

figure;
for i = 1:N
    subplot(N, 1, i);
    plot(t_span, U(1,:,i), 'r');
    hold on;
    plot(t_span, U(2,:,i), 'g');
    plot(t_span, U(3,:,i), 'b');
    legend('u1', 'u2', 'u3');
    title(['OMR ', num2str(i), ' Control Input']);
    xlabel('Time (s)');
    ylabel('Values');
    hold off;
end

figure;
for i = 1:N
    subplot(N, 1, i);
    plot(t_span, State(1,:,i), 'r', t_span, eta_r_vals(1, :), 'b--');
    hold on;
    plot(t_span, State(2,:,i), 'g', t_span, eta_r_vals(2, :), 'm--');
    legend(['x', num2str(i)], 'x_r', ['y', num2str(i)], 'y_r');
    title(['OMR ', num2str(i), ' Trajectory Tracking']);
    xlabel('Time (s)');
    ylabel('States');
    hold off;
end

figure;
for i = 1:N
    subplot(N, 1, i);
    plot(t_span, State(1,:,i)- eta_r_vals(1, :), 'r');
    hold on;
    plot(t_span, State(2,:,i)- eta_r_vals(2, :), 'g');
    legend(['x-error', num2str(i)], ['y-error', num2str(i)]);
    title(['OMR ', num2str(i), ' Trajectory Tracking Error']);
    xlabel('Time (s)');
    ylabel('States');
    hold off;
end

figure;
hold on;
for i = 1:N
    plot(State(1, :,i), State(2, :,i), 'DisplayName', ['OMR ', num2str(i)]);
end
plot(eta_r_vals(1, :), eta_r_vals(2, :), 'k--', 'DisplayName', 'Reference Trajectory');
legend;
xlabel('x');
ylabel('y');
title('Trajectories of Unmanned OMRs');
hold off;


function u = single_vehicle_control_law(t, state,dot_inv_g1, State_multiple, i, g1, g2, f2, eta_r, eta_r_dot, eta_r_ddot, K_eta, K_q, Gamma, Lap, B_comm, P, N, bar_fr, dot_bar_fr, ddot_bar_fr, hatbar_wr)
u = zeros(3, 1);
e1 = zeros(3, 1);
e2 = zeros(3, 1);
alpha1 = zeros(3, 1);
z=zeros(3,1);
dot_z=zeros(3,1);
dot_hatbar_wr = zeros(2,1);
ddot_hatbar_wr = zeros(2,1);
dot_Delt = zeros(3,N);

% Reference trajectory values
eta_r_val = eta_r(t);
eta_r_dot_val = eta_r_dot(t);
eta_r_ddot_val = eta_r_ddot(t);

bar_fr = bar_fr(t);
dot_bar_fr = dot_bar_fr(t);
ddot_bar_fr = ddot_bar_fr(t);

% Local neighborhood consensus error        注意：此处State_multiple(6,4)使用了邻域的一二阶系统状态。
Delt=(State_multiple(1:3,:) -eta_r(t))';
for j=1:N
    dot_Delt(:,j)=g1(State_multiple(:,j))*State_multiple(4:6,j)-eta_r_dot_val;        % 用的并不是实时的邻域状态，而是一个步长的初始状态。
end
dot_Delt=dot_Delt';

Z=(Lap + B_comm)*Delt;
dot_Z=(Lap + B_comm)*dot_Delt;
z(1:3)=Z(i,:)';
dot_z(1:3)=dot_Z(i,:)';
mu=B_comm(i,i);

% Tracking errors
rho=[0,0,0,0;...
    0,0,0,0;...
    0,0,0,0];
% rho=[3,3,6,6;...
%     0,3,0,3;...
%     0,0,0,0];
e1(1:3) = state(1:3) - mu*eta_r_val - (1-mu)*(bar_fr'*hatbar_wr)- rho(:,i);
dot_hatbar_wr(1:2)=-Gamma * bar_fr * z;
ddot_hatbar_wr(1:2)=-Gamma *(dot_bar_fr*z+bar_fr*dot_z);

% Virtual control
alpha1(1:3) = g1(state) \(mu*eta_r_dot_val+(1-mu)*(dot_bar_fr'*hatbar_wr(1:2)+ bar_fr' * dot_hatbar_wr(1:2))-K_eta * P(i,i)*z);

e2(1:3) = state(4:6) - alpha1(1:3);
dot_alpha1 = dot_inv_g1(state) * (mu*eta_r_dot_val+(1-mu)*(dot_bar_fr'*hatbar_wr(1:2)+ bar_fr' * dot_hatbar_wr(1:2))-K_eta * P(i,i)*z)...
    +g1(state)' * (mu*eta_r_ddot_val+(1-mu)*(ddot_bar_fr'*hatbar_wr(1:2)+2*dot_bar_fr'*dot_hatbar_wr(1:2)...
    +bar_fr'*ddot_hatbar_wr(1:2))-K_eta * P(i,i)*dot_z);

% Actual control
u(1:3) = g2 \ (-f2(state) + dot_alpha1 - e1 - K_q * e2);
end

function u = single_vehicle_optimal_control(t, state,dot_inv_g1, State_multiple, i, g1, g2, f2, eta_r, eta_r_dot, eta_r_ddot, K_eta, K_q, Gamma, Lap, B_comm, P, N, bar_fr, dot_bar_fr, ddot_bar_fr, hatbar_wr,W1,R_u,num_weights,alpha_w,Q_e,timeScale)
u = zeros(3+num_weights, 1);
e1 = zeros(3, 1);
e2 = zeros(3, 1);
alpha1 = zeros(3, 1);
z=zeros(3,1);
dot_z=zeros(3,1);
dot_hatbar_wr = zeros(2,1);
ddot_hatbar_wr = zeros(2,1);
dot_Delt = zeros(3,N);

% Reference trajectory values
eta_r_val = eta_r(t);
eta_r_dot_val = eta_r_dot(t);
eta_r_ddot_val = eta_r_ddot(t);

bar_fr = bar_fr(t);
dot_bar_fr = dot_bar_fr(t);
ddot_bar_fr = ddot_bar_fr(t);

% Local neighborhood consensus error        注意：此处State_multiple(6,4)使用了邻域的一二阶系统状态。
Delt=(State_multiple(1:3,:) -eta_r(t))';
for j=1:N
    dot_Delt(:,j)=g1(State_multiple(:,j))*State_multiple(4:6,j)-eta_r_dot_val;        % 用的并不是实时的邻域状态，而是一个步长的初始状态。
end
dot_Delt=dot_Delt';

Z=(Lap + B_comm)*Delt;
dot_Z=(Lap + B_comm)*dot_Delt;
z(1:3)=Z(i,:)';
dot_z(1:3)=dot_Z(i,:)';
mu=B_comm(i,i);

% Tracking errors
rho=[0,0,0,0;...
    0,0,0,0;...
    0,0,0,0];
% rho=[3,3,6,6;...
%     0,3,0,3;...
%     0,0,0,0];
e1(1:3) = state(1:3) - mu*eta_r_val - (1-mu)*(bar_fr'*hatbar_wr)- rho(:,i);
dot_hatbar_wr(1:2)=-Gamma * bar_fr * z;
ddot_hatbar_wr(1:2)=-Gamma *(dot_bar_fr*z+bar_fr*dot_z);

% Virtual control
alpha1(1:3) = g1(state) \(mu*eta_r_dot_val+(1-mu)*(dot_bar_fr'*hatbar_wr(1:2)+ bar_fr' * dot_hatbar_wr(1:2))-K_eta * P(i,i)*z);

e2(1:3) = state(4:6) - alpha1(1:3);
dot_alpha1 = dot_inv_g1(state) * (mu*eta_r_dot_val+(1-mu)*(dot_bar_fr'*hatbar_wr(1:2)+ bar_fr' * dot_hatbar_wr(1:2))-K_eta * P(i,i)*z)...
    +g1(state)' * (mu*eta_r_ddot_val+(1-mu)*(ddot_bar_fr'*hatbar_wr(1:2)+2*dot_bar_fr'*dot_hatbar_wr(1:2)...
    +bar_fr'*ddot_hatbar_wr(1:2))-K_eta * P(i,i)*dot_z);

% Optimal control
dPhi = Derivative_Phi(e2);
actor_update = -0.5 * pinv(R_u) * g2' * dPhi' * W1;
de = f2(state) + g2 * actor_update - dot_alpha1;
sigma2 = dPhi * de;
normsigma2 = sigma2' * sigma2 + 1;
W1Change = alpha_w*(sigma2 / normsigma2^2) * (sigma2' * W1 + e2' *Q_e* e2 + actor_update' * R_u * actor_update);
W1 = W1 - W1Change / timeScale;  

% Actual control
% u(1:3) = alpha1;
u(1:3) = actor_update;
u(4:4+num_weights-1) = W1;
end

function dphi=Derivative_Phi(e)
dphi=[2*e(1),0,0;...
    e(2),e(1),0;...
    e(3),0,e(1);...
    0,2*e(2),0;...
    0,e(3),e(2);...
    0,0,2*e(3)];
end

function [ds] = single_vehicle_dynamics(t, state, control_law, optimal_control, g1,f2,g2)
ds = zeros(6, 1);
u_optimal = optimal_control(t, state);
% u = control_law(t, state) + u_optimal(1:3);
u = u_optimal(1:3);
% u = control_law(t, state);
ds(1:6) = [g1(state) * state(4:6); f2(state) + g2 * u(1:3)];
end

function [hb_wr] = leader_trajectory_estimate_law(t, i, State_multiple, Gamma, eta_r, bar_fr, Lap, B_comm)
hb_wr=zeros(2,1);
z=zeros(3,1);

% Local neighborhood consensus error        注意：此处State_multiple使用了邻域的一阶系统状态。
Delt=(State_multiple(1:3,:) -eta_r(t))';
Z=(Lap + B_comm)*Delt;
z(1:3)=Z(i,:)';

hb_wr(1:2)=-Gamma * bar_fr(t) * z;
end

function [L, B] = communication_topo()
N = 4;
% edges = [2 3; 1 2; 1 4];
edges = [3 2; 2 3; 1 2; 1 4];

% Compute adjacency matrix A
A = zeros(N, N);
for i = 1:size(edges, 1)
    A(edges(i, 2), edges(i, 1)) = 1; % a_ij = 1 if (j, i) is in E
end
% Compute in-degree matrix Delta
Delta = diag(sum(A, 2));
% Compute Laplacian matrix L
L = Delta - A;
% Leader communication matrix B
B = diag([1, 0, 0, 0]);

% Display matrices
disp('Connectivity Matrix A:');
disp(A);
disp('In-degree Matrix Delta:');
disp(Delta);
disp('Laplacian Matrix L:');
disp(L);
disp('Leader Communication Matrix B:');
disp(B);
end
