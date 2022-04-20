clc, clear all
%% Strejcova metoda
t_step = 1;
u_steps = [1;2;-0.7;-5];

for i = 1:length(u_steps)
    u_step = u_steps(i);
    out = sim('model.slx') %start simulation
    y_array(i,:) = out.y_sim(:,2)';
    u_array(i,:) = out.u_sim(:,2)';
end


t = out.tout;
u = out.u_sim(:,2);
y = out.y_sim(:,2);

% figure
% subplot(2,1,1)
% plot(t,y,'LineWidth',1)
% subplot(2,1,2)
% plot(t,u,'LineWidth',1)

figure
subplot(2,1,1)
plot(t,y_array,'LineWidth',1)
subplot(2,1,2)
plot(t,u_array,'LineWidth',1)

%% normalizacia

y_cent = [];
y_norm = [];
for i = 1:length(u_steps)
    y_cent(i,:) = y_array(i,:) - y_array(i,1);
    y_norm(i,:) = y_cent(i,:)/(u_array(i,end)-u_array(i,1));
    u_cent(i,:) = u_array(i,:) - u_array(i,1);
    u_norm(i,:) = u_cent(i,:)/(u_array(i,end)-u_array(i,1));
end

% figure()
% subplot(2,1,1)
% plot(t,y_norm')
% subplot(2,1,2)
% plot(t,u_norm')

%shift
y = mean(y_norm);
u = mean(u_norm);
idx = find(u); %vsetky nenulove prvky
u = u(idx);
y = y(idx);
t = out.tout;
t = t(idx);
t = t-t(1);

figure()
subplot(2,1,1)
plot(t,y','LineWidth',1)
grid on
subplot(2,1,2)
plot(t,u','LineWidth',1)
grid on

% zosilnenie

K = y(end);

% inflexny bod
dy = diff(y');
[inflex,idxx] = max(dy);
t_inflex = t(idxx);
y_inflex = y(idxx);

%% vykreslenie inflexneho bodu
figure()
plot(t,y','LineWidth',1)
hold on
plot(t_inflex,y_inflex,'rx')
grid on
%%
%Výpočet dotyčnice
d = [];
for x = [1:length(y)-1]
    d = [d, (y(x+1)-y(x))/0.1];
end
figure
hold on
[inP, bod] = max(d);
plot(t,y)

%Výpočet dotyčnice
yy = y(bod) + ((y(bod+1)-y(bod))/0.01).*(t-t(bod));
a = y(bod)
b = ((y(bod+1)-y(bod))/0.01);
%Čas nábehu a prieťahu cez rovnicu priamky
TU = (-a/b)+t(bod);
TN =((max(y)-a)/b)+t(bod);

axis([0 30 0 2.1])
plot(t(bod),y(bod), 'ok', 'MarkerFaceColor', 'k', 'MarkerSize', 4)
 
plot([0 30], [y(end),y(end)], '--k')
plot([TU, TU], [y(1), y(end)], '--k')
plot([TN, TN], [y(1), y(end)], '--k')
plot(t, yy,'Linewidth',0.7);

td = t(find(y>0,1)-1);
Tn = TN-TU;
Tu = TU-td;

fn_i=Tu/Tn; 
n = 2;

T=0.3679*Tn
Ds=(fn_i-0.1036)*Tn
D=td+Ds

s = tf('s')
men = K
den=(T*s+1)^n
G=tf(men,den,'Delay',D) %frequency 0.1









