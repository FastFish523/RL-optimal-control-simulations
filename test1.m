rng('default'); % Reset random number generator for reproducibility
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


totalIter = 6000;
timeScale = 300;
num_weights = 8;
num_states = 6;
num_inputs = 6;

Episode=400;
% Episode=350;
alpha1 = 0.1;
alpha2 = 1;
R = 1 / 1;
q = 20;
noise_coef = 0.1;

% 参考轨迹 eta_r q_r
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

% Initialize variables
z = ones(num_states, 1)*0.5;
% W1 = [0.2,0,0,0,0,0,0.2,0,0,0,0,0.2,0,0,0,0.2,0,0,0.2,0,0.2]';
W1 = rand(num_weights,1);
actor_update = zeros(num_inputs,1);
changeW1 = zeros(totalIter, num_weights);
all_W1 = zeros(Episode, num_weights);
all_Z = zeros(totalIter, num_states);
all_E = zeros(totalIter, num_states);
all_u = zeros(Episode,totalIter, num_inputs);
refer = zeros(num_states,totalIter);
mark_Z = zeros(totalIter, num_states);

for episode = 1:Episode
    z = randn(num_states, 1) * 0.5;
    if (episode == 300) || (episode == Episode)
        z = [0.4279;-0.4708;-0.008;0.6727;0.1922;0.0298];
    end
    all_W1(episode, :) = W1';
    for iter = 1:totalIter
        t = iter / timeScale;
        if episode<1
            refer(:,iter)=[0;0;0;0;0;0]';
%             refer(:,iter)=[cos(pi/10*t);sin(pi/10*t);pi/2 + pi/10 * t;0;0;0]';
        else
%             refer(:,iter)=[cos(pi/10*t);sin(pi/10*t);pi/2 + pi/10 * t;all_u(300,iter, 1);all_u(300,iter, 2);all_u(300,iter, 3)]';
                    refer(:,iter)=[eta_r(t);q_r(t)]';
        end
        e=z-refer(:,iter);
        all_Z(iter, :) = z';
        all_E(iter, :) = e';
        dPhi = Derivative_Phi(e);        
        
        % updated policy
        actor_update = - 0.5 * G(z)' * dPhi' * W1;
%         actor_update = actor_update + (rand(size(actor_update)) - 0.5) * noise_coef;

        dz = F(z) + G(z) * actor_update;
        sigma2 = dPhi * dz;
        normsigma2 = sigma2' * sigma2 + 1;

        if (iter==1) || ((all_E(iter,:)*all_E(iter,:)' - all_E(iter-1,:)*all_E(iter-1,:)') < 0)
            W1Change = alpha1*(sigma2 / normsigma2^2) * (sigma2' * W1 + q * Q(e) + actor_update' * R * actor_update);
        else
            %                 W1Change = (sigma2 / normsigma2^2) * (sigma2' * W1 + q * Q(z) + actor' * R * actor);
            W1Change = -0.5* alpha2*dPhi*G(z)*pinv(R)*G(z)'*e + alpha1*(sigma2 / normsigma2^2) * (sigma2' * W1 + q * Q(e) + actor_update' * R * actor_update);
        end

        changeW1(iter, :) = W1Change';
%         if episode>300
%             W1(4:6) = W1(4:6) - W1Change(4:6) / timeScale;
%         else
            W1 = W1 - W1Change / timeScale;
%         end
        all_u(episode,iter, :) = actor_update';
        z = z + dz / timeScale;
    end
end
save("all_u.mat","all_u");
save("refer.mat","refer");
save("all_Z.mat","all_Z");

% Plotting
figure;
plot(1:Episode, all_W1(1:Episode, 1),1:Episode, all_W1(1:Episode, 2), 1:Episode, all_W1(1:Episode, 3),...
    1:Episode, all_W1(1:Episode, 4),1:Episode, all_W1(1:Episode, 5),1:Episode, all_W1(1:Episode, 6),...
    1:Episode, all_W1(1:Episode, 7),1:Episode, all_W1(1:Episode, 8),1:Episode, all_W1(1:Episode, 8),...
    1:Episode, all_W1(1:Episode, 10),1:Episode, all_W1(1:Episode, 11),1:Episode, all_W1(1:Episode, 12),...
    1:Episode, all_W1(1:Episode, 13),1:Episode, all_W1(1:Episode, 14),1:Episode, all_W1(1:Episode, 15),...
    1:Episode, all_W1(1:Episode, 16),1:Episode, all_W1(1:Episode, 17),1:Episode, all_W1(1:Episode, 18),...
    1:Episode, all_W1(1:Episode, 19),1:Episode, all_W1(1:Episode, 20),1:Episode, all_W1(1:Episode, 21));
% legend('W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8','W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16','W17', 'W18', 'W19', 'W20', 'W21');
title('W_C');

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

figure;
until = totalIter;
plot(1:until, all_E(1:until, 1), 1:until, all_E(1:until, 2), 1:until, all_E(1:until, 3),...
    1:until, all_E(1:until, 4),1:until, all_E(1:until, 5),1:until, all_E(1:until, 6)...
    );
title('Tracking Error');
legend('e1', 'e2', 'e3', 'e4', 'e5', 'e6');


trajectory_plot;

function dphi=Derivative_Phi(e)
dphi=[2*e(1),0,0,0,0,0;...
    e(4),0,0,e(1),0,0;...
    0,2*e(2),0,0,0,0;...
    0,e(5),0,0,e(2),0;...
    0,0,2*e(3),0,0,0;...
    0,0,0,2*e(4),0,0;...
    0,0,0,0,2*e(5),0;...
    0,0,0,0,0,2*e(6)];
end

function f = F(z)
global m m_I  C_t R1;
q=[z(4);z(5);z(6)];
f1=[0;0;0];
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
F_theta=[cos(z(3)), -sin(z(3)), 0;...
    sin(z(3)), cos(z(3)), 0;...
    0,         0,         1];
g1=F_theta;
cM=[m,0,0;...
    0,m,0;...
    0,0,m_I];
cB=[1/2,1/2,-1;...
    sqrt(3)/2,-sqrt(3)/2,0;...
    L,L,L];
g2=cM\cB*C_t/R1;
I0=zeros(3,3);
g=[g1,I0;...
   I0,g2];
end

function q = Q(e)
q = e' * e;
end
