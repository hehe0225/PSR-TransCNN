clc; clear all; close all;
%% 50 minute data partitioning training and testing, constructing labels
load('HLANS5km_true.mat');
load('HLANS5km_all_time.mat');
fs=3276.8; % sampling rate
data_rs_h=[];
for i=1:1:length(data)/fs
    data_rs_h(:,i)=data(i:i+fs,1);
end
% HLANorth read front 50min
HLANS5km_time=HLANS5km(1:50);
tline2=0:0.01633333:50;tline2=tline2(1,1:3000);
tline2(1,3000)=50;
xline=HLANS5km_time(1:50,1);
tline=0:1:49;
xline2=interp1(tline,xline,tline2,'PCHIP');
figure;plot(tline2,xline2,'LineStyle','none','Marker','*');

%% Label
tline3=0:0.02777778:50;
tline=0:1:49;
xline3=interp1(tline,xline,tline3,'PCHIP');
figure;plot(tline3,xline3);
temp_label=roundn(xline2,-1);
% Rank
[sort_label,pos_o_label]=sort(temp_label,'ascend');
% Categorize
Label=ones(3000,1);
for i=1:length(sort_label)
    if(sort_label(i)==sort_label(i+1))
        Label(i+1)=Label(i);
    else
        Label(i+1)=Label(i-1)+1;
    end
end
% Label homing
for i=1:length(sort_label)
    label(pos_o_label(i))=Label(i);
end    
data_rs_hT=data_rs_h';
data_rs_hT=data_rs_hT(:,1:3276);
SwellHLANorthS5=[data_rs_hT label'];
% save('./SwellHLANorthS5_all.mat','SwellHLANorthS5');

%% 
% load actual distance
range_50min=xline2';
label_50min=label';
range_label_50=[range_50min label_50min];
% save('range_label_50.mat','range_label_50');

%% plot
% actual
HLANS5km_time=HLANS5km(1:50);
tline2=0:0.01633333:50;tline2=tline2(1,1:3000);
tline2(1,3000)=50;
xline=HLANS5km_time(1:50,1);
tline=0:1:49;
xline2=interp1(tline,xline,tline2,'PCHIP');
figure;plot(tline2,xline2);
hold on;
plot(tline2(1,1:2400),xline2(1,1:2400));