close all;clear;clc;
tic;
computTime=0;
reX=[];
load('./mat/signal_noise.mat');
data=yt;
N=length(data);
max_d=6;%the maximum value of the time delay 200

sigma=std(data);%calcute standard deviation s_d

for t=1:max_d
    t
    s_t=0;
    delt_s_s=0;
    for m=2:5
        s_t1=0;
        for j=1:4
            r=sigma*j/2;
            data_d=disjoint(data,N,t);% Decompose the time series into t disjoint time series
            [ll,N_d]=size(data_d);
            s_t3=0;
            for i=1:t
                %i
                Y=data_d(i,:);
                C_1(i)=correlation_integral(Y,N_d,r);
                M=N_d-(m-1);  % Number of points in phase space
                for ii=1:m      % Dimension of phase space
                    for jj=1:M           
                        X(ii,jj)=Y((ii-1)+jj);
                    end
                end

            end

        end

    end

   
end

save('X_signal_noise.mat','X');

         