load("all_u.mat");
% 提取机器人的x和y坐标数据
x_robot = all_Z(:, 1);  % z中第一列为机器人的x坐标数据
y_robot = all_Z(:, 2);  % z中第二列为机器人的y坐标数据

% 生成参考轨迹的数据
x_reference = refer(1,:);  % 生成参考轨迹的x坐标数据
y_reference = refer(2,:);   % 生成参考轨迹的y坐标数据

% % 生成初始策略轨迹的数据
% W0=[0.2, 0, 0, 0.2, 0, 0.2]';
% all_Z0=initial_trajectory(W0);
% x_inital = all_Z0(:,1);  % 生成初始策略轨迹的x坐标数据
% y_initial = all_Z0(:,2);   % 生成初始策略轨迹的y坐标数据

figure;
% 绘制机器人的轨迹（实线）
plot(x_robot, y_robot, 'b-', 'LineWidth', 1.5);
hold on;  % 保持图形，以便绘制参考轨迹

% 绘制参考轨迹（虚线）
plot(x_reference, y_reference, 'r--', 'LineWidth', 1.5);


% % 绘制初始策略轨迹（点线）
% plot(x_inital, y_initial, 'k:', 'LineWidth', 1.5);

% 添加图例和标签
legend('机器人轨迹', '参考轨迹','初始策略轨迹');
xlabel('X坐标');
ylabel('Y坐标');
title('机器人轨迹');

% 关闭图形保持，以允许下一次绘图
hold off;

function all_z0=initial_trajectory(W0)
totalIter = 6000;
timeScale = 300;
all_z0=zeros(totalIter, 3);
z0=[0.5;0.5;0.5];
for iter = 1:totalIter    
    t = iter / timeScale;
    refer=[cos(pi/10*t);...
        sin(pi/10*t);...
        pi/2 + pi/10 * t];
    %         refer=[0;0;0];
    e=z0-refer;
    all_z0(iter, :) = z0';
    dPhi = Derivative_Phi(e);

    % updated policy
    actor_update = - 0.5 * G(z0)' * dPhi' * W0;

    dz0 = F(z0) + G(z0) * actor_update;
    z0 = z0 + dz0 / timeScale;
end
end

function g = G(z)
g=[cos(z(3)), -sin(z(3)), 0;...
    sin(z(3)), cos(z(3)), 0;...
    0,         0,         1];
end

function dPhi = Derivative_Phi(e)
dPhi = [2 * e(1), 0, 0;...
    e(2), e(1) , 0;...
    e(3), 0, e(1);...
    0, 2 * e(2), 0;...
    0, e(3), e(2);
    0, 0, 2 * e(3)];
end

function q = Q(e)
q = e' * e;
end

function f = F(z)
f=[0;0;0];
end