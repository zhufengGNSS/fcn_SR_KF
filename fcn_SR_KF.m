% 
% Description:
% This file compares three different versions of the Kalman filter.
% The Kalman filter is used for recursive parameter estimation.
% The Kalman filter can handle noisy measurements.
% 
% The first implemented filter (fcn_KF) is the Kalman filter with standard
% update of the covariance matrix P.
% The covariance matrix reflects the uncertainties of the predictions. 
% 
% To improve the numerical stability Potter developed a 
% square root update (fcn_KF_SRP) of the covariance matrix P.
% Another version is the square root covariance update via 
% triangularization (fcn_KF_SRT).
% 
% This file generates a model. Then the three Kalman filters perform an
% estimation of the model parameter. At the end the results are compared.
%
% Sources: 
% Simon, D. (2006): Optimal state estimation
% Kaminski, P. (1971): Discrete Square Root Filtering:
% A Survey of Current Techniques
% Golub, G. (1996): Matrix Computations
%
% Author:
% Markus Knobloch
% uqcot(at)student(dot)kit(dot)edu
% Institut für Fahrzeugsystemtechnik
% Karlsruher Institut für Technologie
% 
% Date: February 10, 2013




function fcn_SR_KF
clear all;
close all;
clc;
format short

% Filter inputs:
% m - Number of samples
% n - Number of parameters
% X - Parameter vector size (n,1)
% F - State transition matrix
% Re - Measurement noise covariance matrix  
% Q - Process/dynamic noise covariance matrix 
% P - Covariance matrix size (n,n)
% S - Square root covariance matrix size (n,n)
% A_M - Input data, size (1,n)
% B_M - Desired signal or output, size (1)
%
% Filter outputs:
% x_new - New state vector
% P_new - New covariance matrix


%--------------------------------------------------------------------------
% Options
%--------------------------------------------------------------------------

% Model options
m=1000; % Number of samples
n=4; % Number of parameters

% Initial true parameter value
X(1,1)=2;
X(2,1)=3;
X(3,1)=10;
X(4,1)=5;

snrB=100; % Signal-to-noise ratio output data 1

snr1=30; % Signal-to-noise ratio input data 1
snr2=50; % Signal-to-noise ratio input data 2
snr3=40; % Signal-to-noise ratio input data 3
snr4=70; % Signal-to-noise ratio input Data 4

Parameter{1}='linear'; %Options parameter 1
Parameter{2}='constant'; %Options parameter 2
Parameter{3}='sin'; %Options parameter 3
Parameter{4}='step'; %Options parameter 4

g=0.002; %Grade
a=1; %Amplitude
f=0.03; %Frequency
s=0.8; %Step-factor

% Filter options
F=eye(n); % State transition matrix
Re=0.95; % Measurement noise covariance matrix
Q=diag([0.5;0.00001;5;0.5]); % Process noise covariance matrix
P_ini=eye(n)*1e10; % Initial covariance matrix

% Initial parameter estimate
% x_ini=X; % true intial parameters
x_ini = ones(n ,1);

%--------------------------------------------------------------------------
% Build the model
%--------------------------------------------------------------------------

[A_meas, B_meas, X]  = build_model(m,n,snrB,snr1,snr2,snr3,snr4,X,Parameter,g,a,f,s);

%--------------------------------------------------------------------------
% Run filter
%--------------------------------------------------------------------------

% Run Kalman filter with standard covariance update

P_old=P_ini;
x_old=x_ini;

x_KF=zeros(m,n); % preallocating

for M=1:m
    
    A_M=A_meas(M,:);
    B_M=B_meas(M);
    
    [x_new, P_new]  = ...
        fcn_KF(x_old, P_old, A_M, B_M,F,Re,Q);
    
    P_old=P_new;
    x_old=x_new;
    
    x_KF(M,:)=x_new';
end

% Run Kalman filter with Potter's square root measurement update

S_old=sqrt(P_ini);
x_old=x_ini;

x_KF_SRP = zeros(m,n); % preallocating

for M=1:m
    
    A_M=A_meas(M,:);
    B_M=B_meas(M);
    
    [x_new, S_new]  = ...
        fcn_KF_SRP(x_old, S_old, A_M, B_M,F,Re,Q);
    
    S_old=S_new;
    x_old=x_new;
    
    x_KF_SRP(M,:)=x_new';
end

% Run Kalman filter with square root covariance update via
% triangularization

S_old=sqrt(P_ini);
x_old=x_ini;

x_KF_SRT=zeros(m,n); % preallocating

for M=1:m
    
    A_M=A_meas(M,:);
    B_M=B_meas(M);
    
    [x_new, S_new]  = ...
        fcn_KF_SRT(x_old, S_old, A_M, B_M,F,Re,Q,n);
    
    S_old=S_new;
    x_old=x_new;
    
    x_KF_SRT(M,:)=x_new';
end

%--------------------------------------------------------------------------
% Compare results
%--------------------------------------------------------------------------

% Calculate error
error_KF=abs((X-x_KF)./X)*100;
mean_KF=mean(error_KF,2);

error_KF_SRP=abs((X-x_KF_SRP)./X)*100;
mean_KF_SRP=mean(error_KF_SRP,2);

error_KF_SRT=abs((X-x_KF_SRT)./X)*100;
mean_KF_SRT=mean(error_KF_SRT,2);

% Accumulated arror
Error_Sum=zeros(1,3);
for M=1:m
    
    Error_Sum(M+1,1)=Error_Sum(M,1)+ mean_KF(M);
    Error_Sum(M+1,2)=Error_Sum(M,2)+ mean_KF_SRP(M);
    Error_Sum(M+1,3)=Error_Sum(M,3)+ mean_KF_SRT(M);
    
end

% Plot results
fcn_plot(X,x_KF,x_KF_SRP,x_KF_SRT,Error_Sum,n,m)

end %% of function=========================================================


function [A_meas, B_meas,X]  = build_model(m,n,snrB,snr1,snr2,snr3,snr4,X,Parameter,g,a,f,s)

%Create input A
A=unifrnd(-20,20,m,n); %Random A

%create Parameter X
X=X';
for N=1:n
    
    
    if strcmp(Parameter{N},'constant') %Constant
        
        for M=1:m
            X(M,N)=X(1,N);
        end
        
    elseif strcmp(Parameter{N},'linear') %Linear
        
        for M=1:m
            X(M,N)=X(1,N)+g*M;
        end
        
    elseif strcmp(Parameter{N},'sin') %Sin
        
        for M=1:m
            X(M,N)=X(1,N)+a*sin(M*f);
        end
        
    elseif strcmp(Parameter{N},'step') %Step
        
        for M=1:((m/2)-1)
            X(M,N)=X(1,N);
        end
        
        
        for M=(m/2):m
            X(M,N)=X(1,N)*s;
        end
    end
end

%Output B
B=zeros(m,1); %preallocating
for M=1:m
    for N=1:n
        B(M)=B(M)+A(M,N)*X(M,N);
    end
end

%%Add noise
A_meas(:,1) = awgn(A(:,1),snr1);
A_meas(:,2) = awgn(A(:,2),snr2);
A_meas(:,3) = awgn(A(:,3),snr3);
A_meas(:,4) = awgn(A(:,4),snr4);

B_meas = awgn(B,snrB);
end

function [x_new, P_new]  = ...
    fcn_KF(x_old, P_old, A_M, B_M,F,Re,Q)
%Kalman filter
%Source: Simon, D. (2006): Optimal state estimation, Chapter 5 ,Page 128/129.

%Prediction - A priori estimate

x_new=F*x_old;
P_new = F * P_old * F';

%Correction - A posteriori estimate

e = B_M - (A_M * x_new);

K = (P_new * A_M') / (Re + A_M*P_new*A_M');

P_new = P_new - K*A_M * P_new + Q;

x_new = x_new + K * e;
end

function [x_new,S_new]  = ...
    fcn_KF_SRP(x_old,S_old,A_M,B_M,F,Re,Q)
%Kalman Filter with Potter's square root measurement update
%Source: Simon, D. (2006): Optimal state estimation, Chapter 6 ,Page 166.
%And: Kaminski, P. (1971): Discrete Square Root Filtering:
%A Survey of Current Techniques, Page 729. 

%Prediction - A priori estimate

x_old=F*x_old;
S_old =F*S_old*F';

%Correction - A posteriori estimate

e = B_M - (A_M * x_old);

Qsr = chol(Q,'lower');
Rsr = chol(Re,'lower');

f=S_old'*A_M';
beta=Rsr+f'*f;
alpha=1/(beta+sqrt(Rsr*beta));
    
K=S_old*f/beta;
    
S_new=S_old-alpha*S_old*(f*f');  

x_new = x_old + K * e;

S_new = fcn_modGramSmith([S_new';Qsr']);
S_new = S_new';
end

function [x_new,S_new]  = ...
    fcn_KF_SRT(x_old,S_old,A_M,B_M,F,Re,Q,n)
%Kalman Filter with square root Covariance update via
%triangularization
%Source: Simon, D. (2006): Optimal state estimation, Chapter 6, Page 169.
%And: Kaminski, P. (1971): Discrete Square Root Filtering:
%A Survey of Current Techniques, Page 731. 


%Prediction - A priori estimate

x_old=F*x_old;
S_old =F* S_old *F';

%Correction - A posteriori estimate

e = B_M - (A_M * x_old);

Qsr = chol(Q,'lower');
Rsr = chol(Re,'lower');

prearray = [Rsr               , zeros(1,n);...
    (S_old'*A_M') , S_old'];
postarray = fcn_house(prearray); %or just %[~,postarray]= qr(prearray);
Kk = postarray(1,2:(n+1))';
r = postarray(1,1);
S_new = postarray(2:(n+1),2:(n+1))';
K = (Kk/r);
x_new = x_old + K * e;
S_new = fcn_modGramSmith([S_new';Qsr']);
S_new = S_new';
end

function W = fcn_modGramSmith(A)
%Source: Simon, D. (2006): Optimal state estimation, Chapter 6 ,Page 172.
[~,n] = size(A);
W = zeros(n);
for k=1:n
    sigma = sqrt(A(:,k)'*A(:,k));
    for j=1:n
        if j==k
            W(k,j)=sigma;
        elseif j==k+1 || j==n
            W(k,j)=(A(:,k)'*A(:,j))/sigma;
        else
            W(k,j)=0;
        end
    end
end

end % of function==========================================================

function A = fcn_house(A)
% source: Golub, G. (1996): Matrix Computations, Chapter 5, Page 211.
[m,n] = size (A);
for j=1:(n-1)
    [v, beta] = fcn_householderVector(A(j:m,j));
    A(j:m , j:n) = (eye(m-j+1) - beta*(v*v')) * A(j:m , j:n);
end
end % of function==========================================================

function [v, beta] = fcn_householderVector(x)
% source: Golub, G. (1996): Matrix Computations, Chapter 5, Page 210.
n = length(x);
sigma = x(2:n)'*x(2:n);
v=[1;...
    x(2:n)];
if sigma == 0
    beta =0;
else
    mu = sqrt(x(1)^2+sigma);
    if x(1) <= 0
        v(1)= x(1) - mu;
    else
        v(1) = -sigma / (x(1) + mu);
    end
    beta = 2*v(1)^2 / (sigma + v(1)^2);
    v = v / v(1);
end
end % of function

function []=fcn_plot(X,x_KF,x_KF_SRP,x_KF_SRT,Error_Sum,n,m)

%Plot result Kalman filter with standard covariance update

x_p=zeros(m,(2*n)); %preallocating
for k=1:n
    x_p(:,k)=X(:,k);
    x_p(:,k+4)=x_KF(:,k);
end
% Create plot
figure
plot_KF=plot(x_p);

h = legend('X 1','X 2','X 3','X 4','X_KF 1','X_KF 2','X_KF 3','X_KF 4',-1);
set(h,'Interpreter','none');
title('Kalman filter with standard covariance update')

set(plot_KF(1),'LineWidth',1,'Color',[0 0 1],'DisplayName','X 1');
set(plot_KF(2),'LineWidth',1,'Color',[0 1 0],'DisplayName','X 2');
set(plot_KF(3),'LineWidth',1,'Color',[1 0 0],'DisplayName','X 3');
set(plot_KF(4),'LineWidth',1,'Color',[0 0 0],'DisplayName','X 4');

set(plot_KF(5),'Marker','x','LineWidth',0.5,'Color',[0 0 1],'DisplayName','X_KF 1');
set(plot_KF(6),'Marker','x','LineWidth',0.5,'Color',[0 1 0],'DisplayName','X_KF 2');
set(plot_KF(7),'Marker','x','LineWidth',0.5,'Color',[1 0 0],'DisplayName','X_KF 3');
set(plot_KF(8),'Marker','x','LineWidth',0.5,'Color',[0 0 0],'DisplayName','X_KF 4');

xlabel('Sample','FontSize',11);
ylabel('Parameter','FontSize',11);


%Plot result Kalman filter with Potter's square root measurement update
for k=1:n
    x_p(:,k)=X(:,k);
    x_p(:,k+4)=x_KF_SRP(:,k);
end
% Create plot
figure
plot_KF=plot(x_p);

h = legend('X 1','X 2','X 3','X 4','X_KF 1','X_KF 2','X_KF 3','X_KF 4',-1);
set(h,'Interpreter','none');
title('Kalman filter with Potter''s square root measurement update')

set(plot_KF(1),'LineWidth',1,'Color',[0 0 1],'DisplayName','X 1');
set(plot_KF(2),'LineWidth',1,'Color',[0 1 0],'DisplayName','X 2');
set(plot_KF(3),'LineWidth',1,'Color',[1 0 0],'DisplayName','X 3');
set(plot_KF(4),'LineWidth',1,'Color',[0 0 0],'DisplayName','X 4');

set(plot_KF(5),'Marker','x','LineWidth',0.5,'Color',[0 0 1],'DisplayName','X_KF_SRP 1');
set(plot_KF(6),'Marker','x','LineWidth',0.5,'Color',[0 1 0],'DisplayName','X_KF_SRP 2');
set(plot_KF(7),'Marker','x','LineWidth',0.5,'Color',[1 0 0],'DisplayName','X_KF_SRP 3');
set(plot_KF(8),'Marker','x','LineWidth',0.5,'Color',[0 0 0],'DisplayName','X_KF_SRP 4');

xlabel('Sample','FontSize',11);
ylabel('Parameter','FontSize',11);


%Plot Kalman filter with square root covariance update via
%triangularization
for k=1:n
    x_p(:,k)=X(:,k);
    x_p(:,k+4)=x_KF_SRT(:,k);
end
% Create plot
figure
plot_KF_SRT=plot(x_p);

h = legend('X 1','X 2','X 3','X 4','X_KF_SRT 1','X_KF_SRT 2','X_KF_SRT 3','X_KF_SRT 4',-1);
set(h,'Interpreter','none');
title('Kalman filter with square root covariance update via triangularization');

set(plot_KF_SRT(1),'LineWidth',1,'Color',[0 0 1],'DisplayName','X 1');
set(plot_KF_SRT(2),'LineWidth',1,'Color',[0 1 0],'DisplayName','X 2');
set(plot_KF_SRT(3),'LineWidth',1,'Color',[1 0 0],'DisplayName','X 3');
set(plot_KF_SRT(4),'LineWidth',1,'Color',[0 0 0],'DisplayName','X 4');

set(plot_KF_SRT(5),'Marker','x','LineWidth',0.5,'Color',[0 0 1],'DisplayName','X_KF_SRT 1');
set(plot_KF_SRT(6),'Marker','x','LineWidth',0.5,'Color',[0 1 0],'DisplayName','X_KF_SRT 2');
set(plot_KF_SRT(7),'Marker','x','LineWidth',0.5,'Color',[1 0 0],'DisplayName','X_KF_SRT 3');
set(plot_KF_SRT(8),'Marker','x','LineWidth',0.5,'Color',[0 0 0],'DisplayName','X_KF_SRT 4');

xlabel('Sample','FontSize',11);
ylabel('Parameter','FontSize',11);

%Plot accumulated error
figure
plot_Error=plot(Error_Sum);

h = legend('KF','KF_SRP','KF_SRT',-1);
set(h,'Interpreter','none');
title('Accumulated error');

set(plot_Error(1),'LineWidth',1,'Color',[1 0 0],'DisplayName','KF');
set(plot_Error(2),'Marker','x','LineWidth',1,'Color',[0 1 0],'DisplayName','KF_SRP');
set(plot_Error(3),'LineWidth',1,'Color',[0 0 1],'DisplayName','KF_SRT');

xlabel('Sample','FontSize',11);
ylabel('Accumulated error','FontSize',11);
end