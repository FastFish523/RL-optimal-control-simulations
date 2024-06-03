clear;
close all;
%% 仿真时间参数
t_span = linspace(0, 20, 6000);
timeScale=300;
Episode = 300;

%% 更新律设计参数
K1=diag([1,1,1]);
R_u=diag([1,1,1]);
Q_e=diag([20,20,20]);
alpha_w=100;
alpha_e=1;
noise_coef=0.1;

num_weights = 6;
num_states = 3;
num_inputs = 3;

%% 系统动态模型
Ct = 0.3;
R = 0.0635;
L = 0.14;
m = 5;
mI = 0.1125;

M = diag([m, m, mI]);
F_theta = @(theta) [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];
C = @(omega) [0, m*omega, 0; -m*omega, 0, 0; 0, 0, 0];
B = [1/2, 1/2, -1; sqrt(3)/2, -sqrt(3)/2, 0; L, L, L];
g1 = @(state) F_theta(state(3));
f2 = @(state) -M \ (C(state(3)) * state(1:3));
g2 = M \ B * Ct / R;
F = @(state) [0;0;0;f2(state)];
I0=zeros(3,3);
G = @(state) [g1(state),I0;...
              I0,g2];


% 简单运算
dot_inv_g1=@(state) [-sin(state(3))*state(6), cos(state(3))*state(6), 0; -cos(state(3))*state(6), -sin(state(3))*state(6), 0; 0, 0, 0];


%% 参考轨迹 eta_r q_r
x_r= @(t) 5*sin(2*t);
y_r= @(t) 5*sin(t);
theta_r= @(t) 0*t;

dot_x_r= @(t) 10*cos(2*t);
dot_y_r= @(t) 5*cos(t);
dot_theta_r= @(t) 0;
ddot_x_r= @(t) -20*sin(2*t);
ddot_y_r= @(t) -5*sin(t);
ddot_theta_r= @(t) 0;

u_r= @(t) cos(theta_r(t))*dot_x_r(t) + sin(theta_r(t))*dot_y_r(t);
v_r= @(t) -sin(theta_r(t))*dot_x_r(t) + cos(theta_r(t))*dot_y_r(t);
omega_r= @(t) dot_theta_r(t);

dot_u_r= @(t) cos(theta_r(t))*ddot_x_r(t) + sin(theta_r(t))*ddot_y_r(t)...
    + cos(theta_r(t))*dot_theta_r(t)*dot_y_r(t) - sin(theta_r(t))*dot_theta_r(t)*dot_x_r(t);
dot_v_r= @(t) cos(theta_r(t))*ddot_y_r(t) - sin(theta_r(t))*ddot_x_r(t)...
    - cos(theta_r(t))*dot_theta_r(t)*dot_x_r(t) - sin(theta_r(t))*dot_theta_r(t)*dot_y_r(t);
dot_omega_r= @(t) ddot_theta_r(t);

eta_r = @(t)  [x_r(t); y_r(t); theta_r(t)];
q_r = @(t)  [u_r(t);v_r(t);omega_r(t)];
% dot_q_r = @(t) [dot_u_r(t);dot_v_r(t);dot_omega_r(t)];
dot_q_r = @(t) [sin(t);cos(t);0];

%% 初始化实际轨迹
eta_actual = zeros(3, length(t_span));
eta_actual(:,1) = [0,0,0];


%% 数值积分计算实际轨迹
for i = 2:length(t_span)
    dt = t_span(i) - t_span(i-1);
    F_theta = [cos(eta_actual(3,i-1)), -sin(eta_actual(3,i-1)), 0;
        sin(eta_actual(3,i-1)),  cos(eta_actual(3,i-1)), 0;
        0, 0, 1];

    alpha1 = q_r(t_span(i));
    eta_dot = F_theta * alpha1;
    eta_actual(:,i) = eta_actual(:,i-1) + eta_dot * dt;
end

eta_plot = eta_r(t_span);

% figure;
% plot(eta_plot(1,:),eta_plot(2,:),'r',eta_actual(1,:),eta_actual(2,:),'g--');
% plot(t,y_r(1,:),'r',t,y_r(2,:),'g',t,eta_actual(1,:),'r--',t,eta_actual(2,:),'g--');

%% 系统状态与网络权重初始化
z2 = [1;1;0];

hat_W=ones(6,1)*0.5;
% hat_W=rand(6,1);
% 用于保存所有数据的参数
all_hat_W=zeros(num_weights,Episode);
all_e2=zeros(num_states,length(t_span));
all_u=zeros(num_states,length(t_span));
value_func=zeros(Episode,1);


%% 强化学习最优控制
all_E = ones(length(t_span), num_states);

for episode = 1 : Episode
%     z2 = [0.5;0.5;0.5];
    z2 = randn(num_states, 1)*0.5;
    for steps = 1:length(t_span)
        t=t_span(steps);
        if episode < 200
            e=z2-[0;0;0];
        else
            e=z2-[q_r(t)];
        end
        all_E(steps, :) = z2';
        dPhi=Derivative_Phi(e);
        u=-0.5*g2'*dPhi'*hat_W;
        %         if episode <200
        %             u= u+(exp(-0.008*t)) * noise_coef;
        %         end
        dz2=f2(z2)+g2*u;
        %                 de=g1(e2)*u;
        %         de=f2(e2)+g2*u-dot_q_r(t);
        sigma2 = dPhi * dz2;
        normsigma2 = sigma2' * sigma2 + 1;

        if (steps==1) || ((all_E(steps,:)*all_E(steps,:)' - all_E(steps-1,:)*all_E(steps-1,:)') < 0)
            dot_hat_W=alpha_w*(sigma2 / normsigma2^2) * (sigma2' * hat_W + e'*Q_e*e + u' * R_u * u);
        else
            dot_hat_W=-0.5* alpha_e*dPhi*g2*pinv(1)*g2'*e + alpha_w*(sigma2 / normsigma2^2) * (sigma2' * hat_W + e'*Q_e*e + u' * R_u * u);
        end
        hat_W=hat_W-dot_hat_W/timeScale;
        z2=z2+dz2/timeScale;

        % 保存数据
        if steps == length(t_span)
            all_hat_W(:,episode) =hat_W;
        end
        all_e2(:,steps) = z2;
        all_u(:,steps) = u;
        value_func(episode,1) = value_func(episode,1) + z2'*Q_e*z2 + u'*u;
    end
    %     set(w1_plot_handles,'XData',1:episode,'YData',all_hat_W(:,length(t_span),1:episode));
    %     drawnow;
end

%% 画图
figure;
plot(1:Episode,all_hat_W(:,1:Episode));

figure;
plot(t_span,all_e2);

figure;
plot(1:Episode,value_func);


%% Functions
function dphi=Derivative_Phi(e)
dphi=[2*e(1),0,0;...
    e(2),e(1),0;...
    e(3),0,e(1);...
    0,2*e(2),0;...
    0,e(3),e(2);...
    0,0,2*e(3)];
end


% function dphi=Derivative_Phi(e)
% dphi=[2*e(1),0,0,0,0,0;...
%     e(2),e(1),0,0,0,0;...
%     e(3),0,e(1),0,0,0;...
%     e(4),0,0,e(1),0,0;...
%     e(5),0,0,0,e(1),0;...
%     e(6),0,0,0,0,e(1);...
%     0,2*e(2),0,0,0,0;...
%     0,e(3),e(2),0,0,0;...
%     0,e(4),0,e(2),0,0;...
%     0,e(5),0,0,e(2),0;...
%     0,e(6),0,0,0,e(2);...
%     0,0,2*e(3),0,0,0;...
%     0,0,e(4),e(3),0,0;...
%     0,0,e(5),0,e(3),0;...
%     0,0,e(6),0,0,e(3);...
%     0,0,0,2*e(4),0,0;...
%     0,0,0,e(5),e(4),0;...
%     0,0,0,e(6),0,e(4);...
%     0,0,0,0,2*e(5),0;...
%     0,0,0,0,e(6),e(5);...
%     0,0,0,0,0,2*e(6)];
% end
