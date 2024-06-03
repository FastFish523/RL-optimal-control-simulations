% 时间向量
t = linspace(0, 10, 2000); 

% 初始化变量
u_r = zeros(1, length(t));
v_r = zeros(1, length(t));
omega_r = 0.1 * ones(1, length(t));

% 角度 theta 随时间变化，这里假设 theta 是一个函数
theta = 2 * t; % 假设 theta(t) = 0.2 * t (仅为示例，可以根据实际情况调整)

load("all_Alpha1.mat");
% % 计算 q_r
% for i = 1:length(t)
%     A = [cos(theta(i)), -sin(theta(i));
%          sin(theta(i)),  cos(theta(i))];
%     b = [1; cos(t(i))];
%     
%     % 求解 [u_r; v_r]
%     qr = A \ b;
%     u_r(i) = qr(1);
%     v_r(i) = qr(2);
% end
% 
% q_r = [u_r; v_r; omega_r];
q_r = Alpha1(:,:,1);
% 初始化实际轨迹
eta_actual = zeros(3, length(t));

% 初始位置
eta_actual(:,1) = [0; 0; 0]; % 初始位置可以根据需要调整

% 数值积分计算实际轨迹
for i = 2:length(t)
    dt = t(i) - t(i-1);
    F_theta = [cos(eta_actual(3,i-1)), -sin(eta_actual(3,i-1)), 0;
               sin(eta_actual(3,i-1)),  cos(eta_actual(3,i-1)), 0;
               0, 0, 1];
    eta_dot = F_theta * q_r(:,i-1);
    eta_actual(:,i) = eta_actual(:,i-1) + eta_dot * dt;
end

% 参考轨迹 eta_r
eta_r = [5*t; 5*sin(t); 0*t+1];

% 绘制结果
figure;
subplot(3,1,1);
plot(t, eta_r(1,:), 'r--', t, eta_actual(1,:), 'b');
title('x over time');
xlabel('Time (s)');
ylabel('x');
legend('Reference', 'Actual');

subplot(3,1,2);
plot(t, eta_r(2,:), 'r--', t, eta_actual(2,:), 'b');
title('y over time');
xlabel('Time (s)');
ylabel('y');
legend('Reference', 'Actual');

subplot(3,1,3);
plot(t, eta_r(3,:), 'r--', t, eta_actual(3,:), 'b');
title('theta over time');
xlabel('Time (s)');
ylabel('theta');
legend('Reference', 'Actual');
