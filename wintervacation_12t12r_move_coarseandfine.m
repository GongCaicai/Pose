%% 12.01 最佳方法，每帧处理时间降低到55秒，且内存需求降低
%% 11.30 直接找三维最大值中心点，并确定三维矩阵来确定人的位置
%% 11.30 粗粒度可以更大些，细粒度范围可以更小一些
%% 11.28 先粗粒度定位，再细粒度beamforming，做人体姿态估计
%% 11.24 换一种算法，争取提高运算效率
%% 11.23 用循环提高分辨率
%% 11.17 用学诚的采数据的方式采集的，12t12r
%% 11.17 与第一帧原始数据做差分
clear all
close all
clc;

%% 前期对采集到的原始数据预处理

csi1 = struct2array(load('1216_stayhome\12.16\100k_12t12r_401_outwall_move2.mat')).data;%11.4 2560*500=80*16*2*500
timestamp = struct2array(load('1216_stayhome\12.16\100k_12t12r_401_outwall_move2.mat')).timestamp;
[fc, f_step, halfband] = deal(2.8e9, 1.0e7, 2e9);
[fStep, c, da, de] = deal(fc-halfband:f_step:fc+halfband, 3e8, 0.032, 0.032);
%11.2 da为接收天线的物理间距，de为发射天线的物理间距
[S_NUM, S_POINTS, NUM_RX, NUM_TX, MAX_TAR] = deal(200, 401, 12, 12, 1);
samplevnt = 20.26 / 500;
cntpic=0;
n_frames = S_NUM;%11.4 共500帧
K_index = 1:S_POINTS;
R_index = 1:NUM_RX*NUM_TX;
csi = complex(csi1(1:2:size(csi1,1)-1,:),csi1(2:2:size(csi1,1),:));
csi_wifi = zeros(length(K_index) * length(R_index),S_NUM);%1.4 28944*300 = 201*144*300
for i=1:size(csi,2)
    tempcsi = reshape(csi(:,i),S_POINTS,NUM_RX*NUM_TX);%1.4 变为201行 * 144列，即一个通道对应201个频点
    tempwifi = tempcsi(K_index,R_index);
    csi_wifi(:,i) =  reshape(tempwifi.' ,[],1);%1.4 变为144*201，然后按照特定频点，144通道来
end
csi = csi_wifi;
%% 11.2 发射端 的位置
N = NUM_TX * NUM_RX;
txPos = zeros(2,NUM_TX);
txPos(1,:) = [1:1:NUM_TX] * de;
txPos(1,:) = txPos(1,:) - 6.5*de;

%% 11.2 接收端 的位置
rxPos = zeros(2,NUM_RX);
rxPos(2,:) = [1:1:NUM_RX] * da;
rxPos(2,:) = rxPos(2,:) - rxPos(2,NUM_RX)-0.155;
rxPos = flip(rxPos,2);%1.8 新增，rx1->rx12，沿z轴从上到下
%% 11.2 生成包含收发天线位置和所有频点信息的矩阵
[virTxPos_X,virRxPos_X,fSteps] = ndgrid(txPos(1,:),rxPos(1,:),fStep); %10.2生成12*12*80的矩阵
[virTxPos_Y,virRxPos_Y,fSteps] = ndgrid(txPos(2,:),rxPos(2,:),fStep); 
%% 11.2 作图，都只处理了第一个矩阵（共length（fStep）=80个矩阵）,因为每个矩阵相同，画出天线的空间分布
figure(1);
plot(reshape(virTxPos_X(:,:,1),[],1),reshape(virTxPos_Y(:,:,1),[],1),'g*')
hold on;
plot(reshape(virRxPos_X(:,:,1),[],1),reshape(virRxPos_Y(:,:,1),[],1),'r*')
ylabel('z-axis');
xlabel('x-axis');
axis equal;
title('天线分布图');

%% 11.2 构造三维平面，获得x y z的范围，应该是指的空间中各个目标点的位置


% x_grid = -0.7:0.047:0.71;%11.2  31长度
% y_grid = 0.2:0.05:1.45;%11.2  26长度
% z_grid = -0.5: 0.04 :0.7;  %11.2  31长度

% 11.17
% x_grid = -1.4:0.094:1.42;%11.2  31长度
% y_grid = 0.4:0.1:2.9;%11.2  26长度
% z_grid = -0.6: 0.05 :1;  %11.2  31长度
% 11.23
% x_grid = -4:0.05:4;%11.2  30长度
% y_grid = 0:0.05:2.05;%11.2  25长度
% z_grid = -1.5:0.05:1.5;  %11.2  26长度
% 11.17
% x_grid = -3:0.2:3;%11.2  30长度
% y_grid = -0.2:0.06:1.3;%11.2  25长度
% z_grid = -0.8:0.08:1.2;  %11.2  26长度
% 11.28
% x_grid_coarse = -4:0.2:4;  %11.28        41
% y_grid_coarse = 0:0.1:4;  %11.28         41
% z_grid_coarse = -1.5:0.1:1.5;  %11.28    31
% delta_x = 1.2;%11.28 range为2.4米
% delta_y = 0.9;%11.28 range为1.8米
% delta_z = 1.2;%11.28 range为2.4米
% 11.30
x_grid_coarse = -3:0.15:3;%11.2  30长度
y_grid_coarse = 0:0.2:5;%11.2  25长度
z_grid_coarse = -1.5:0.15:2.5;  %11.2  26长度
delta_x = 1.14; %11.28 range为2.28米
delta_y = 0.91; %11.28 range为1.82米
delta_z = 1.225;%11.28 range为2.45米
lenx = 39;
leny = 27;
lenz = 36;

[spaceXyzGrid] = generateSpaceGrid(x_grid_coarse, y_grid_coarse, z_grid_coarse);%11.4生成31z*31x*26y，其中，y的每个31*31矩阵中的每个值为y的其中一个值，共26个y值，故共26个矩阵
%11.2 构造出x,y,z的三维空间平面，为3*length(y)*length(x)*length(z)=3*24986的二维矩阵，具体见函数注释
[roundTripBfGrid] = generategVN_NearFiledGrid(virTxPos_X,virTxPos_Y, virRxPos_X, virRxPos_Y, fSteps, spaceXyzGrid);
% file_name = ['roundTripBfGrid_temp' '.mat'];
% save(file_name, 'roundTripBfGrid');
bfWindow = zeros(length(z_grid_coarse)*length(x_grid_coarse) *length(y_grid_coarse),1);

%% 11.2 计算出空间每个位置与收发天线位置的距离并算出相位差，保存在roundTripBfGrid中

pic_num = 1;
folder='F:\winter_vacation\data_move_close_away\三视图\'; %%定义变量
folder1='F:\winter_vacation\data_move_close_away\俯视图\';
folder2='F:\winter_vacation\data_move_close_away\正视图\';
folder3='F:\winter_vacation\data_move_close_away\左视图\';
if exist(folder)==0 %%判断文件夹是否存在
    mkdir(folder);  %%不存在时候，创建文件夹
    mkdir(folder1);
    mkdir(folder2);
    mkdir(folder3);
end
csi_temp = diff(csi,1,2);
for i = 122:n_frames - 1

    raw_scan_all = csi_temp(:,i);%11.25 减去前一帧
    currData = reshape(raw_scan_all.',[NUM_TX,NUM_RX,S_POINTS]);
    aoatofout = currData(:).' * roundTripBfGrid;%11.2 不同位置的beamforming
    bfWindow(:,end) = aoatofout.';%11.2 把最后一列替换为aoatofout.'  
    absMeanSig = mean(abs((bfWindow.')),1);%11.2 返回每一列的均值,200*24986->1*24986,对200帧aoatofout取平均
    absMeanSigMatrix =  reshape(absMeanSig, [length(z_grid_coarse) length(x_grid_coarse) length(y_grid_coarse)]);%11.2 31*31*26
    %% 11.30 粗粒度确定人体中心位置（也就是最大反射中心）
    human_center = max(absMeanSigMatrix(:));
    [center_z,center_x,center_y] = find3d(absMeanSigMatrix == human_center);
    range_x_center = x_grid_coarse(1) + (center_x - 1) * (x_grid_coarse(2) - x_grid_coarse(1));
    range_y_center = y_grid_coarse(1) + (center_y - 1) * (y_grid_coarse(2) - y_grid_coarse(1));
    range_z_center = z_grid_coarse(1) + (center_z - 1) * (z_grid_coarse(2) - z_grid_coarse(1));
    range_x_min = range_x_center - delta_x;
    range_y_min = range_y_center - delta_y;
    range_z_min = range_z_center - delta_z;
    range_x_max = range_x_center + delta_x;
    range_y_max = range_y_center + delta_y;
    range_z_max = range_z_center + delta_z;
    %% 11.30 细粒度人体框定范围
    x_grid_fine = range_x_min:0.06:range_x_max;  %11.30 
    y_grid_fine = range_y_min:0.07:range_y_max;  %11.30
    z_grid_fine = range_z_min:0.07:range_z_max;  %11.30
    [spaceXyzGrid_fine] = generateSpaceGrid(x_grid_fine, y_grid_fine, z_grid_fine);
    aoatofout_fine = [];
    for n = 1:leny/9
        [roundTripBfGrid_fine_temp] = generategVN_NearFiledGrid(virTxPos_X,virTxPos_Y, virRxPos_X, virRxPos_Y, fSteps, spaceXyzGrid_fine(:,[(n-1)*lenx*lenz*9+1:1:n*lenx*lenz*9]));
        aoatofout_fine_temp = currData(:).' * roundTripBfGrid_fine_temp;%11.2 不同位置的beamforming
        aoatofout_fine = [aoatofout_fine aoatofout_fine_temp];
    end
        
    bfWindow_fine = zeros(length(z_grid_fine)*length(x_grid_fine) *length(y_grid_fine),1);   
    bfWindow_fine(:,end) = aoatofout_fine.';%11.2 把最后一列替换为aoatofout.'  
    absMeanSig_fine = mean(abs((bfWindow_fine.')),1);%11.2 返回每一列的均值,200*24986->1*24986,对200帧aoatofout取平均
    absMeanSigMatrix_fine =  reshape(absMeanSig_fine, [length(z_grid_fine) length(x_grid_fine) length(y_grid_fine)]);%11.2 31*31*26    
    
    
    displayMatXY = gather(squeeze(max(absMeanSigMatrix,[],1)));%11.2 计算每一列的最大值，变为1行（行压缩），31*26
    displayMatXY_fine = gather(squeeze(max(absMeanSigMatrix_fine,[],1)));%11.2 计算每一列的最大值，变为1行（行压缩），31*26
    fprintf('i=%d, XY=%f, XY_fine=%f\n',i,displayMatXY(1,1),displayMatXY_fine(1,1));
    
    displayMatXZ = gather(squeeze(max(absMeanSigMatrix,[],3)));%11.2 计算每一单位的最大值，单位压缩，31*31
    displayMatXZ_fine = gather(squeeze(max(absMeanSigMatrix_fine,[],3)));%11.2 计算每一单位的最大值，单位压缩，31*31
    fprintf('i=%d, XZ=%f, XZ_fine=%f\n',i,displayMatXZ(2,2),displayMatXZ_fine(2,2));
    
    displayMatYZ = gather(squeeze(max(absMeanSigMatrix,[],2)));%11.2 计算每一行的最大值，变为一列（列压缩），31*26 
    displayMatYZ_fine = gather(squeeze(max(absMeanSigMatrix_fine,[],2)));%11.2 计算每一行的最大值，变为一列（列压缩），31*26 
    fprintf('i=%d, YZ=%f, YZ_fine=%f\n',i,displayMatYZ(3,3),displayMatYZ_fine(3,3));
  
    %% 11.30作图
    %TopView
    figure(2);
    imagesc(y_grid_coarse,x_grid_coarse, displayMatXY);%11.2 分别表示横坐标和纵坐标
    rectangle('Position',[range_y_min range_x_min 2*delta_y 2*delta_x],'EdgeColor','r','LineWidth',4);
    xlabel('y-axis');
    ylabel('x-axis');
    title('俯视图——粗粒度',num2str(i));
    view(-90,90)
    F11=getframe(gcf);
    imwrite(F11.cdata,['F:\winter_vacation\data_move_close_away\俯视图\','coarse',int2str(i),'.jpg']);
  
    figure(3);
    imagesc(y_grid_fine,x_grid_fine, displayMatXY_fine);
    xlabel('y-axis');
    ylabel('x-axis');
    title('俯视图——细粒度',num2str(i));
    view(-90,90)
    F12=getframe(gcf);
    imwrite(F12.cdata,['F:\winter_vacation\data_move_close_away\俯视图\','fined',int2str(i),'.jpg']);
    
    %FrontView
    figure(4);
    imagesc(x_grid_coarse, z_grid_coarse, displayMatXZ); 
    rectangle('Position',[range_x_min range_z_min 2*delta_x 2*delta_z],'EdgeColor','r','LineWidth',4);
    xlabel('x-axis');
    ylabel('z-axis');
    title('正视图——粗粒度',num2str(i));
    view(0,-90)
    F21=getframe(gcf);
    imwrite(F21.cdata,['F:\winter_vacation\data_move_close_away\正视图\','coarse',int2str(i),'.jpg']);
    
    figure(5);
    imagesc(x_grid_fine, z_grid_fine, displayMatXZ_fine); 
    xlabel('x-axis');
    ylabel('z-axis');
    title('正视图——细粒度',num2str(i));
    view(0,-90)
    F22=getframe(gcf);
    imwrite(F22.cdata,['F:\winter_vacation\data_move_close_away\正视图\','fined',int2str(i),'.jpg']);
%     
    %LeftView
    figure(6);
    imagesc(y_grid_coarse, z_grid_coarse, displayMatYZ);
    rectangle('Position',[range_y_min range_z_min 2*delta_y 2*delta_z],'EdgeColor','r','LineWidth',4);
    xlabel('y-axis');
    ylabel('z-axis');
    title('左视图——粗粒度',num2str(i));
    view(0,-90)
    F31=getframe(gcf);
    imwrite(F31.cdata,['F:\winter_vacation\data_move_close_away\左视图\','coarse',int2str(i),'.jpg']);

    figure(7);
    imagesc(y_grid_fine, z_grid_fine, displayMatYZ_fine);
    xlabel('y-axis');
    ylabel('z-axis');
    title('左视图——细粒度',num2str(i));
    view(0,-90)
    F32=getframe(gcf);
    imwrite(F32.cdata,['F:\winter_vacation\data_move_close_away\左视图\','fined',int2str(i),'.jpg']);

end