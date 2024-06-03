function OMR_verified()
%rng('default'); % Reset random number generator for reproducibility
close all;

% Initialize parameters
global m m_I L C_t R1;
m=5;
m_I=0.1125;
L=0.14;
C_t=0.3;
R1=0.0635;
cM=[m,0,0;...
    0,m,0;...
    0,0,m_I];
cB=[1/2,1/2,-1;...
    sqrt(3)/2,-sqrt(3)/2,0;...
    L,L,L];
K2=cM\cB*C_t/R1;

load('all_u.mat');
load('refer.mat');
totalIter = 6000;
timeScale = 300;
Episode=400;
num_states=6;

% Initialize variables
% z = ones(num_states, 1);
% z = [0;0;0;0;0;0];
z = [0;0;0;0.6727;0.1922;0.0298];
actor_velocity(:,:) = all_u(Episode,:,1:3);
actor_torque(:,:) = all_u(Episode,:,4:6);

TORQUE=1;

for iter = 1:totalIter
    t = iter / timeScale;
    all_Z(iter, :) = z';
    
    if TORQUE
        dz = F(z) + G(z) * actor_torque(iter,:)';
        z = z + dz / timeScale;
    else
        F_theta=[cos(z(3)), -sin(z(3)), 0;...
            sin(z(3)), cos(z(3)), 0;...
            0,         0,         1];
        dz = [F_theta * actor_velocity(iter,:)';0;0;0];
        z = z + dz / timeScale;
    end
end

% Plotting

figure;
plot(1:totalIter,all_u(Episode,1:totalIter,1),1:totalIter,all_u(Episode,1:totalIter,2),1:totalIter,all_u(Episode,1:totalIter,3),...
    1:totalIter,all_u(Episode,1:totalIter,4),1:totalIter,all_u(Episode,1:totalIter,5),1:totalIter,all_u(Episode,1:totalIter,6));
legend('u1', 'u2', 'u3','u4', 'u5', 'u6');
title('u');


figure;
until = totalIter;
plot(1:until, all_Z(1:until, 1), 1:until, all_Z(1:until, 2), 1:until, all_Z(1:until, 3),...
    1:until, all_Z(1:until, 4),1:until, all_Z(1:until, 5),1:until, all_Z(1:until, 6)...
    );
title('System States');
legend('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6');


% 提取机器人的x和y坐标数据
x_robot = all_Z(:, 1);  % z中第一列为机器人的x坐标数据
y_robot = all_Z(:, 2);  % z中第二列为机器人的y坐标数据
% 生成参考轨迹的数据
x_reference = refer(1,:);  % 生成参考轨迹的x坐标数据
y_reference = refer(2,:);   % 生成参考轨迹的y坐标数据

figure;
% 绘制机器人的轨迹（实线）
plot(x_robot, y_robot, 'b-', 'LineWidth', 1.5);
hold on;  % 保持图形，以便绘制参考轨迹
% 绘制参考轨迹（虚线）
plot(x_reference, y_reference, 'r--', 'LineWidth', 1.5);

% 添加图例和标签
legend('机器人轨迹', '参考轨迹','初始策略轨迹');
xlabel('X坐标');
ylabel('Y坐标');
title('机器人轨迹');
% 关闭图形保持，以允许下一次绘图
hold off;
end

function dphi=Derivative_Phi(e)
dphi=[2*e(1),0,0,0,0,0;...
    e(2),e(1),0,0,0,0;...
    e(3),0,e(1),0,0,0;...
    e(4),0,0,e(1),0,0;...
    e(5),0,0,0,e(1),0;...
    e(6),0,0,0,0,e(1);...
    0,2*e(2),0,0,0,0;...
    0,e(3),e(2),0,0,0;...
    0,e(4),0,e(2),0,0;...
    0,e(5),0,0,e(2),0;...
    0,e(6),0,0,0,e(2);...
    0,0,2*e(3),0,0,0;...
    0,0,e(4),e(3),0,0;...
    0,0,e(5),0,e(3),0;...
    0,0,e(6),0,0,e(3);...
    0,0,0,2*e(4),0,0;...
    0,0,0,e(5),e(4),0;...
    0,0,0,e(6),0,e(4);...
    0,0,0,0,2*e(5),0;...
    0,0,0,0,e(6),e(5);...
    0,0,0,0,0,2*e(6)];
end

function f = F(z)
global m m_I  C_t R1;
q=[z(4);z(5);z(6)];
F_theta=[cos(z(3)), -sin(z(3)), 0;...
    sin(z(3)), cos(z(3)), 0;...
    0,         0,         1];
f1=F_theta*q;
cM=[m,0,0;...
    0,m,0;...
    0,0,m_I];
cC=[0,m*z(6),0;...
    -m*z(6),0,0;...
    0,0,0];
f2=-cM\cC*q;
f=[f1;f2];
end

function g = G(z)
global m m_I L C_t R1;
cM=[m,0,0;...
    0,m,0;...
    0,0,m_I];
cB=[1/2,1/2,-1;...
    sqrt(3)/2,-sqrt(3)/2,0;...
    L,L,L];
g2=cM\cB*C_t/R1;
I0=zeros(3,3);
g=[I0;...
   g2];
end

function q = Q(e)
q = e' * e;
end
