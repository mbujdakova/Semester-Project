clc, clear all

%% CONSTANTS
k11 = 0.8;
k22 = 0.87;
F1 = 0.75;
F2 = 1.40;
q01s = 0.7;
q02s = 0.4;
% k11 = 0.55;
% k22 = 0.95;
% F1 = 0.8;
% F2 = 1.35;
% q01s = 0.7;
% q02s = 0.4;
%% STEADY-STATE
h1s = (q01s/k11)^2;
h2s = ((q01s + q02s)/k22)^2;
k1 = k11/(2*sqrt(h1s));
k2 = k22/(2*sqrt(h2s));

%% STATE SPACE
A = [-k1/F1 0; k1/F2 -k2/F2];
B = [1/F1 0; 0 1/F2];
C = [1 0; 0 1];
D = [0 0;0 0];
[num,den] = ss2tf(A,B,C,D,1);
out =sim('tanks_students_2020_assign')
%%
t=out.u_lin(:,1);
u1_lin=out.u_lin(:,2);
u2_lin=out.u_lin(:,3);

u1_nelin=out.u_nelin(:,2);
u2_nelin=out.u_nelin(:,3);
h1=out.h1(:,2);
h2=out.h2(:,2);
x1_lin=out.x_lin(:,2);
x2_lin=out.x_lin(:,3);

%%
n=11;
%% Comparison
figure()
subplot(2,1,1)
plot(t,h1,t,x1_lin+h1s,'LineWidth',1)
      xlabel('$t \ [\mathrm{s}]$'),ylabel('$h_1 \ [\mathrm{m}]$'),;
        
        axis([0 480 0.4 1.2])
        subplot(2,1,2)
plot(t,h2,t,x2_lin+h2s,'LineWidth',1)
      xlabel('$t \ [\mathrm{s}]$'),ylabel('$h_2 \ [\mathrm{m}]$'),;

        xlim([0 480])
%% LINEAR MODEL
for i = 1:n+1
    idx(i) = round(length(x1_lin)/(n+1))*i;
    if i > 1
        t_array(:,i-1) = t(idx(i-1):idx(i));
        u_array(:,i-1) = u1_lin(idx(i-1):idx(i));
        u2_array(:,i-1) = u2_lin(idx(i-1):idx(i));
        
        x1_array(:,i-1) = x1_lin(idx(i-1):idx(i));
        x1_array_centr(:,i-1) = x1_array(:,i-1) - x1_array(1,i-1);
        x1_array_norm(:,i-1) = x1_array_centr(:,i-1) / (u_array(end,i-1) - u_array(1,i-1));
        
        x2_array(:,i-1) = x2_lin(idx(i-1):idx(i));
        x2_array_centr(:,i-1) = x2_array(:,i-1) - x2_array(1,i-1);
        x2_array_norm(:,i-1) = x2_array_centr(:,i-1) / (u_array(end,i-1) - u_array(1,i-1));
    end
    
    figure(1)
    hold on
    if i == 1
        subplot(2,1,1)
        plot(t,x1_lin+h1s,'r','LineWidth',1);
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$x_1$'),;
        
        axis([0 480 -0.4 0.4])
        box on
    else
        subplot(2,1,1)
        plot(t_array(:,i-1),x1_array(:,i-1)+h1s,'LineWidth',1);
    end
    
    figure(1)
    hold on
    if i == 1
        subplot(2,1,2)
        plot(t,x2_lin,'r');
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$x_2$');
        xlim([0 480]) 
        box on
    else
        subplot(2,1,2)
        plot(t_array(:,i-1),x2_array(:,i-1),'LineWidth',1);
        xlim([0 480])
    end
      figure(20)
      
    hold on
    if i == 1
        subplot(2,1,1)
        plot(t,u1_lin,'r');
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$u_1$');
        xlim([0 480]) 
        box on
    else
        subplot(2,1,1)
        plot(t_array(:,i-1),u_array(:,i-1),'LineWidth',1);
        axis([0 480 -0.2 0.2])
    end
          figure(20)
    hold on
    if i == 1
        subplot(2,1,2)
        plot(t,u2_lin,'r');
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$u_2$');
        xlim([0 480]) 
        
        box on
    else
        subplot(2,1,2)
        plot(t_array(:,i-1),u2_array(:,i-1),'LineWidth',1);
        xlim([0 480])
    end
end
%% VISUALIZATION OF NORM. RESPONSES
figure(3)
tt=0:0.01:40;
plot(tt,x1_array_norm,'b','LineWidth',1);
 xlabel('$t \ [\mathrm{s}]$'),ylabel('$x_1$');
axis([0 40 0 2.5])
box on
figure(4)
plot(tt,x2_array_norm,'r','LineWidth',1);
axis([0 40 0 3.3])
 xlabel('$t \ [\mathrm{s}]$'),ylabel('$x_2$');
box on
%%
for i = 1:n+1
    idx(i) = round(length(h1)/(n+1))*i;
    if i > 1
        h1_array(:,i-1) = h1(idx(i-1):idx(i));
        h1_array_centr(:,i-1) = h1_array(:,i-1) - h1_array(1,i-1);
        h1_array_norm(:,i-1) = h1_array_centr(:,i-1) / (u_array(end,i-1) - u_array(1,i-1));
        u_array(:,i-1) = u1_nelin(idx(i-1):idx(i));
        u2_array(:,i-1) = u2_nelin(idx(i-1):idx(i));
        
        h2_array(:,i-1) = h2(idx(i-1):idx(i));
        h2_array_centr(:,i-1) = h2_array(:,i-1) - h2_array(1,i-1);
        h2_array_norm(:,i-1) = h2_array_centr(:,i-1) / (u_array(end,i-1) - u_array(1,i-1));
    end
    
    figure(5)
    hold on
    if i == 1
        subplot(2,1,1)
        plot(t,h1,'r','LineWidth',1);
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$h_1 \ [\mathrm{m}]$'),;
        
        axis([0 480 0.4 1.2])
        box on
    else
        subplot(2,1,1)
        plot(t_array(:,i-1),h1_array(:,i-1),'LineWidth',1);
    end
    
    figure(5)
    hold on
    if i == 1
        subplot(2,1,2)
        plot(t,h2,'r');
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$h_2 \ [\mathrm{m}]$');
        xlim([0 480]) 
        box on
    else
        subplot(2,1,2)
        plot(t_array(:,i-1),h2_array(:,i-1),'LineWidth',1);
        xlim([0 480])
    end
          figure(6)
    hold on
    if i == 1
        subplot(2,1,1)
        plot(t,u1_nelin,'r');
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$q_1 \ [\mathrm{m^3/s}]$');
        xlim([0 480]) 
        box on
    else
        subplot(2,1,1)
        plot(t_array(:,i-1),u_array(:,i-1),'LineWidth',1);
        axis([0 480 0.5 0.9])
    end
          figure(6)
    hold on
    if i == 1
        subplot(2,1,2)
        plot(t,u2_nelin,'r');
        xlabel('$t \ [\mathrm{s}]$'),ylabel('$q_2 \ [\mathrm{m^3/s}]$');
        xlim([0 480]) 
        box on
    else
        subplot(2,1,2)
        plot(t_array(:,i-1),u2_array(:,i-1),'LineWidth',1);
        xlim([0 480])
    end
    
end

%%
figure(7)
tt=0:0.01:40;
plot(tt,h1_array_norm);
h1p=mean(h1_array_norm')
hold on
plot(tt,h1p,'b','LineWidth',1.5)
 xlabel('$t \ [\mathrm{s}]$'),ylabel('$h_1 \ [\mathrm{m}]$');
axis([0 40 0 2.5])
box on
figure(8)
h2p=mean(h2_array_norm');
plot(tt,h2_array_norm);
hold on
plot(tt,h2p,'r','LineWidth',1.5)
axis([0 40 0 3.3])
 xlabel('$t \ [\mathrm{s}]$'),ylabel('$h_2 \ [\mathrm{m}]$');
box on
%% First order
h1_mean = mean(h1_array_norm')';
u_mean = u_array(:,1)./u_array(:,1);
t_mean = t_array(:,1);

idx = find(u_mean > 0);
t = t_mean(idx) - t_mean(idx(1));
u = u_mean(idx);
y1 = h1_mean(idx);
K3 = y1(end);

t31 = t(find(y1>0.2*K3,1));
t32 = t(find(y1>0.8*K3,1));
yt31 = y1(find(y1>0.2*K3,1));
yt32 = y1(find(y1>0.8*K3,1));

M = [log(1-yt31/K3) -1; log(1-yt32/K3) -1];
a = [-t31; -t32];
x = M\a;

T3 = x(1);
D3 = x(2);
table(K3,T3,D3)
y_ident3 = K3*(1 - exp(-(tt(2:4001) - D3)/T3));
figure(9)
hold on
plot(tt,y1,'b','LineWidth',1);
plot(tt(1:4000),y_ident3,'--r','LineWidth',1);
plot([t31 t32],[yt31 yt32],'ko','MarkerSize',5,'MarkerFaceColor','k');
ylabel('$h_1 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')
yline(y1(end),'--')
axis([0 40 0 2.5])
box on
xt = [t31+1 t32+1 1];
yt = [yt31 yt32 y1(end)+0.05];
str = {'$[t_1, y(t_1)]$','$[t_2, y(t_2)]$','$K$'}
text(xt,yt,str)
hold off

suma2=sum((y_ident3-y1(2:4001)').^2);
RMSE2=sqrt(suma2/length(y_ident3))


RMSEa=sqrt(sum(mean(y_ident3-y1(2:4001)').^2))

%% HIGHER ORDER
idx0 = find(y1>0,1)-1;
dy = diff(y1);
dt = diff(t);
d = dy./dt;
idxI = find(d == max(d),1);

tI = t(idxI);
tI1 = t(idxI + 1);
yI = y1(idxI);
yII = y1(idxI + 1);
a = (yI - yII)/(tI - tI1);
b = yII - a*tI1;
t1 = -b/a;
t2 = (K3 - b)/a;
yt1 = 0;
yt2 = K3;
Tu = t1 - t(idx0);
Tn = t2 - t1;

fn_v = Tu/Tn %vypocitane
n = 1
gn = 1
fn =0

T3H = gn*Tn;
D3H = (fn_v-fn)*Tn + tt(idx0);

T3 = table(K3,T3H,D3H,n)

suma = 0;
for i=0:n-1
    suma = suma + 1/factorial(i)*((t - D3H)/T3H).^(i);
end
y1_ident3 = K3*(1 - exp(-(t - D3H)/T3H) .* suma ); 
idx = find(t<D3H);
y1_ident3(idx) = zeros(size(idx));
rmse3H = sqrt((mean(y1 - y1_ident3).^2))

suma3=sum((y1_ident3(2:4001)-y1(2:4001)).^2);
RMSE3=sqrt(suma3/length(y1_ident3))

figure(10)
hold on
plot(t,y1,'b','LineWidth',1);
plot(t(1:4000),y_ident3,'--r','LineWidth',2);
plot(t,y1_ident3,'-.m','LineWidth',1);
ylabel('$h_1 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')
axis([0 40 0 2.5])
box on
hold off
%%
figure(11)
hold on
plot(t,y1,'m','LineWidth',1)
plot(t,y1_ident3,'--b','LineWidth',1);
% plot(t(idx0),y1(idx0),'y*')
% plot(t(idxI),y1(idxI),'r*')
plot([t1 t2],[yt1 yt2],'ko','MarkerSize',5,'MarkerFaceColor','k')
plot([t1 t2],[yt1 yt2],'k')
yline(K3,'--k')
ylabel('$h_1 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')

box on
axis([0 40 0 2.5])
%% Second tank
%% 1st ORDER
h2_mean = (mean(h2_array_norm'))';
u_mean = u_array(:,1)./u_array(:,1);
t_mean = t_array(:,1);

% figure(12)
% hold on
% plot(h2_mean,'--','LineWidth',2)
% box on

idx = find(u_mean > 0);
t = t_mean(idx) - t_mean(idx(1));
u = u_mean(idx);
y2 = h2_mean(idx);

% figure
% plot(t,u);
% figure
% plot(t,y2);

K4 = y2(end);

t41 = t(find(y2>0.2*K4,1));
t42 = t(find(y2>0.9*K4,1));
yt41 = y2(find(y2>0.2*K4,1));
yt42 = y2(find(y2>0.9*K4,1));

M = [log(1-yt41/K4) -1; log(1-yt42/K4) -1];
a = [-t41; -t42];
x = M\a;

T4 = x(1);
D4 = x(2);
table(K4,T4,D4)

y_ident4 = K4*(1 - exp(-(t - D4)/T4));
idx = find(t<D4);
y_ident4(idx) = zeros(size(idx));

figure(13)
hold on
plot(t,y2,'b','LineWidth',1);
plot(t,y_ident4,'r','LineWidth',1);
yline(K4,'--');
ylabel('$h_2 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')
plot(t41,yt41,'ko','MarkerSize',5,'MarkerFaceColor','k');
plot(t42,yt42,'ko','MarkerSize',5,'MarkerFaceColor','k');
axis([0 40 0 3.5])
xt = [t41+1 t42+1 1];
yt = [yt41 yt42-0.05 y2(end)+0.1];
str = {'$[t_1, y(t_1)]$','$[t_2, y(t_2)]$','$K$'}
text(xt,yt,str)
hold off
sumah2=sum((y_ident4-y2).^2);
RMSEh2=sqrt(sumah2/length(y_ident4))
figure
plot(t,u)
%% HIGHER ORDER
idx0 = find(y2>0,1)-1;
dy = diff(y2);
dt = diff(t);
d = dy./dt;
idxI = find(d == max(d),1);

tI = t(idxI); %inflexny bod
tI1 = t(idxI + 1); %pomocny bod na ziskanie a a b
yI = y2(idxI);
yII = y2(idxI + 1);
a = (yI - yII)/(tI - tI1); %y=ax+b
b = yII - a*tI1;
t1 = -b/a; % Toto posuvas smerom doprava (xova os)
t2 = (K4 - b)/a; % Alternativne menit
yt1 = 0;
yt2 = K4;
Tu = t1 - t(idx0);
Tn = t2 - t1;

fn_v = Tu/Tn %vypocitane
% n = input('Zadaj hodnotu n z tabulky (ak fn_v > 0.104 < 0.218 daj 2; inak daj 1):') 
% gn = input('Zadaj hodnotu gn z tabulky (ak fn_v > 0.104 < 0.218 daj 0.368; inak daj 1):')
% fn = input('Zadaj hodnotu fn z tabulky (ak fn_v > 0.104 < 0.218 daj 0.104; inak daj 0):')
n=1;
gn=1;
fn=0;
T4H = gn*Tn;
D4H = (fn_v-fn)*Tn + t(idx0);

T4 = table(K4,T4H,D4H,n)
%% STREJC
figure(15)
hold on
plot(t,y2,'r','LineWidth',1)
yline(K4,'--k')
plot([t1 t2],[yt1 yt2],'ko','MarkerSize',3,'MarkerFaceColor','k')
plot([t1 t2],[yt1 yt2],'k')
plot(t(idxI),y2(idxI),'ro','MarkerSize',5,'MarkerFaceColor','r')
axis([0 40 0 3.3])
ylabel('$h_2 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')

%%
suma = 0;
for i=0:n-1
    suma = suma + 1/factorial(i)*((t - D4H)/T4H).^(i);
end
y2_ident4 = K4*(1 - exp(-(t - D4H)/T4H) .* suma ); 
% idx = find(t<D4H);
% y2_ident4(idx) = zeros(size(idx));

figure(16)
hold on
plot(t,y2,'b','LineWidth',1)
% plot(t,y2_ident4)

rmse4H = sqrt((((y2 - y2_ident4)/length(y2)).^2))

s=tf('s');
G=tf(K4,[T4H 1],'InputDelay',D4H);
[vyst cas]=step(G,0:0.01:40);
plot(cas,vyst,'k','LineWidth',1)
axis([0 40 0 3.3])
ylabel('$h_2 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')

sumah2s=sum((vyst-y2).^2);
RMSEh2s=sqrt(sumah2s/length(vyst)) %0.2370

%%
figure
hold on
plot(t,y2,'b','LineWidth',1)
s=tf('s');
G1=tf(K4,(conv([T4H+0.02 1],[T4H+0.02 1])),'InputDelay',0.01);
[vyst1 cas1]=step(G1,0:0.01:40);
plot(cas1,vyst1,'r','LineWidth',1)
axis([0 40 0 3.3])
ylabel('$h_2 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')
plot(cas,vyst,'k','LineWidth',1)
sumah2sup=sum((vyst1-y2).^2);
RMSEh2sup=sqrt(sumah2sup/length(vyst1))
%% Comparision
figure()
hold on
plot(t,y_ident4,'m','LineWidth',1)
plot(t,y2,'b','LineWidth',1)
[vyst cas]=step(G,0:0.01:40);
plot(tgg,ygg,'r','LineWidth',1)
plot(tg,yg_v(:,1),'k','LineWidth',1)
axis([0 40 0 3.3])
ylabel('$h_2 \ [\mathrm{m}]$')
xlabel('$t \ [\mathrm{s}]$')
plot(t,y_ident4,'m','LineWidth',1)