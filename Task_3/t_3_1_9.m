%1 +
%2 +
%3 +
%4 +
%5 ?
%6 ?
%7 
%8 +
%9 +
%10
%%
%Task #3_1
%Оценка следует из следствия из теоремы Лейбница: abs(S - S_n) < b_n+1
f = @(n) ((-1).^n)./(n.^2);
S = -(pi.^2)./12;
n = 50;
S_n_m = zeros(1,n);
S_n = 0;
S_d = zeros(1,n);
for i = 1:n
    S_n = S_n + f(i);
    S_n_m(i) = S_n;
    S_d(i) = f(i+1);
end    
%plot(1:n,S - S_n_m,1:n,S_d);
plot(1:n,abs(S - S_n_m),1:n,abs(S_d));
legend('S - S_n','b_n+1');

%%
%Task #3_2
x = linspace(-2*pi,2*pi,100);
f = @(x) cos(x) - x./pi;
hold on
plot(x,cos(x),x,x./pi);
axis equal;
[x,y] = ginput();
for i = 1:numel(x)
   x_s = fzero(f,x(i));
   plot(x_s,cos(x_s),'r.','MarkerSize',20);
   disp('Корень:');
   disp(x_s);
   disp('Невязка:');
   disp(abs(cos(x_s) - cos(x(i))));
end    
hold off
%%
%Task #3_3
eps = 0.00001;
f = @(x) (abs(x) > eps).*x.*sin(1./x);
a = 3;
n = 100;
x = linspace(-a,a,n);
x_s_m = zeros(1,n);
for i = 1:n
    if abs(x(i)) > 1
    x_s_m(i) = (x(i)/abs(x(i))).*0.3183;    
        
    else    
    x_s_m(i) = fzero(f,x(i));

    end    
end    
    plot(x,x_s_m);
%%
%Task 3_4'
t0 = 0;
t1 = 1;
b = -1/2;
c = 1/3;
speed_x = 10;
speed_y = 10;

alpha = 1;
f = @(x) (1 - (x).^2 ).^(1/2);
x = linspace(-1,1,100);
hold on 
plot(x,f(x),'r',x,-f(x),'r');
axis equal;
hold off
flag = 1;
nFrames = 200;
mov(1:nFrames) = struct('cdata',[],'colormap',[]);
v_s_b = [speed_x - b, speed_y - c];

while flag 
    options = odeset('Events', @myEventsFcn);
    [t, y, te, ye, ie]  = ode45(@(t,y) odefcn(t,y),[t0 t1],[b c speed_x speed_y],options);
    n = numel(y(:,1));
    y_x = y(:,1);
    y_y = y(:,2);
    
    for i = 1:n
        %hold on
        plot(x,f(x),'r',x,-f(x),'r',y_x(i),y_y(i),'r.','MarkerSize',20);
        axis equal;
        %hold off
        % plot(y_x(i),y_y(i),'r.','MarkerSize',20);
        %hold off
        mov(i) = getframe();
    end
    
   
   if ((size(te))~=0)
        t0 = te;
   else
        t0 = t1;
        flag = 0;
   end
   if (size(ye)~=0)
       y = @(x) (ye(2)/ye(1))*x;
       eps_x = 0.0001;
       x_eps = ye(1) - (ye(1)/abs(ye(1)))*eps_x;
       y_eps = y(x_eps);
       b = x_eps;
       c = y_eps;
       e_1 = [ye(1), ye(2)];
       e_2 = [-ye(2), ye(1)];
       
       v_s = e_2*dot(e_2,v_s_b) - e_1*dot(e_1,v_s_b)./alpha;
       speed_x = v_s(1);
       speed_y = v_s(2);
       %v_s_b = [speed_x, speed_y];
       v_s_b = v_s;
   end    
       
end
   
    
    
   

%%
%Task #3_4
%по-другому направить скорость.
t0 = 0;
t1 = 2;
b = -1/3;
c = 1/3;
speed_x = 100;
speed_y = 1;
spdy = speed_y;
alpha = 5;
f = @(x) (1 - (x).^2 ).^(1/2);
x = linspace(-1,1,100);
hold on 
plot(x,f(x),'r',x,-f(x),'r');
axis equal;
hold off
flag = 1;
spd = (speed_x.^2 + speed_y.^2).^(1/2);
nFrames = 200;
mov(1:nFrames) = struct('cdata',[],'colormap',[]);
l = 0;
while flag 
    options = odeset('Events', @myEventsFcn);
    [t, y, te, ye, ie]  = ode45(@(t,y) odefcn(t,y),[t0 t1],[b c speed_x speed_y],options);
    n = numel(y(:,1));
    y_x = y(:,1);
    y_y = y(:,2);
    
    for i = 1:n
        %hold on
        plot(x,f(x),'r',x,-f(x),'r',y_x(i),y_y(i),'r.','MarkerSize',20);
        axis equal;
        %hold off
        % plot(y_x(i),y_y(i),'r.','MarkerSize',20);
        %hold off
        mov(i) = getframe();
    end
    
   %disp(ye);
   %disp(t0); 
   if ((size(te))~=0)
        t0 = te;
   else
        t0 = t1;
        flag = 0;
   end
   if (size(ye)~=0)
        
        if (l == 0)
            v_1 = ye(1) - b;
            v_2 = ye(2) - c;
        else
            v_1 = ye(1) - b1;
            v_2 = ye(2) - c1;
        end    
        l = 1;
        v_1_n = ye(2)/ye(1);
        v_2_n = 1;
        angle =  calAngleBetweenVectors([v_1,v_2],[v_1_n,v_2_n]);
        b1 = ye(1);
        c1 = ye(2);
        if ((angle > 0.01) && (angle < 3.13) && (angle < 1.55) && (angle > 1.57))
            disp(angle);
            disp('k');
            y = @(x) (ye(2)/ye(1))*x;
            eps_x = 0.0001;
            x_eps = ye(1) - (ye(1)/abs(ye(1)))*eps_x;
            y_eps = y(x_eps);
            b = x_eps;
            c = y_eps;
            %b = ye(1) - (ye(1))*0.0001./abs(ye(1));
            %c = ye(2) - (ye(2))*0.0001./abs(ye(2));
            spd = spd/alpha;
            speed_x = -(cos(angle)/abs(cos(angle)))*spd*ye(1);
            %speed_x = -(cos(angle)/abs(cos(angle)))*speed_x/alpha;
            speed_y = -(sin(angle)/abs(sin(angle)))*speed_y;
            %speed_y = -(sin(angle)/abs(sin(angle)))*spd*ye(2);
        else
            y = @(x) (ye(2)/ye(1))*x;
            spd = spd/alpha;
            eps_x = 0.001;
            x_eps = ye(1) - (ye(1)/abs(ye(1)))*eps_x;
            y_eps = y(x_eps);
            b = x_eps;
            c = y_eps;
            %b = ye(1) - (ye(1))*0.0001./abs(ye(1));
            %c = ye(2) - (ye(2))*0.0001./abs(ye(2));
            speed_x = -spd*ye(1);
            speed_y = -spd*ye(2);
            
        end    
       
   end   
       
    
    
    % speed_x =  -spd*(ye(1) - 0.001);
   % speed_y =  spd*(ye(2) - 0.001);
    
   if abs(spd) < 0.01
        disp('flag_exit');
        flag = 0; 
    end    
end

%%
%Task #3_5
alpha = 1;%1 1 0  1 кф рождаемости жертв
gamma = 1;%0 1 1  1 кф убыли хищников
beta = 1;%1 3 3  1 кф убийств жертв
delta = 0;%1 5 1  0 кф воспроизводства хищников
tspan = [0 2];
y0 = [1,2]; %[1 2]
[t,y] = ode45(@(t,y) odefcn_5(t,y,alpha,beta,gamma,delta),tspan,y0);
y_t = @(t) exp(exp(t) - t - 1)*2;
x_t = @(t) exp(t);

hold on

plot(t,y(:,1),t,y(:,2));
plot3(t,y(:,1),y(:,2));
%disp(y(:,1) - x_t(t));
%plot(t,y_t(t),t,x_t(t));
legend('жертва','хищник','integral curve','хищник аналит.','жертва аналит.');
hold off
view(3);

%%
%Task #3_6
a11 = 2;  %2 k -1 c -2 f -2 s 1 dk
a12 = 1;   %1    2    5    1   0  
a21 = 1;  %1   -1   -1    1   0 
a22 = 1;   %1    1    1    1   1
n = 60;
t = linspace(0, 2*pi,n);

r = 1/2;
%1)[0 10] n = 60 r = 1/2 2)[0 10] как в 3 3)[0 10] n = 25 r = 1/2 4)[0 10]
%n = 50 r = 1/2 5) [0 10] n = 40 r = 1/2
hold on           
for i = 1:n
   
    
       
           y0 = r.*[cos(t(i)),sin(t(i))];
           tspan = [0 10];
    
           [t,y] = ode45(@(t,y) odefcn_6(t,y,a11,a12,a21,a22),tspan,y0);
           n_q = numel(y(:,1));
  
           
    
           step = 2;
           n_q = fix(n_q/step)*step;
    
            for l = 1:step:n_q - step
                 leng = ((y(l - 1 + step,1) - y(l,1)).^2 + (y(l - 1 + step,2) - y(l,2)).^2).^(1/2);
                 q = quiver(y(l,1),y(l,2),(y(l - 1 + step,1) - y(l,1))/leng,(y(l - 1 + step,2) - y(l,2))/leng,'Color','blue'); 
                 %q.LineWidth = 1;
            end    
   
            plot(y(:,1),y(:,2),'r');
      
     
  
    
   
end
axis([-2 2 -2 2]);
hold off
%%

%Task #3_7_1'
clc
clear
hold on
n = 10;
xx = linspace(-5,5);
yy = linspace(-5,5);
[x_1, y_1] = meshgrid(xx,yy);
u = y_1 - x_1 +x_1.*y_1;
v = x_1 - y_1 - x_1.^2 - y_1.^3;
q = quiver(x_1,y_1,u,v);
x_p = linspace(0,2*pi,n);
axis equal;
for i = 1:n
y0 = [5*cos(x_p(i)),5*sin(x_p(i))];
tspan = [-10 10];
[t, y] = ode45(@(t,y) odefcn_7_11(t,y),tspan,y0);
plot(y(:,1),y(:,2));



end

hold off

%%
%Task #3_7_1'

n = 10;
xx = linspace(-3,3);
yy = linspace(-3,3);
[x_1, y_1] = meshgrid(xx,yy);
u = y_1 - x_1 +x_1.*y_1;
v = x_1 - y_1 - x_1.^2 - y_1.^3;
hold on
%q = quiver(x_1,y_1,u,v);
hold off
x_p = linspace(0,2*pi - 0.001,n);
axis equal;
r = 1;
for i = 1:n
   y0 = r.*[cos(x_p(i)),sin(x_p(i))];
   r = r + 1;
   tspan = [0 1];
   [t, y] = ode45(@(t,y) odefcn_7_1(t,y),tspan,y0);
   hold on 
   plot(y(:,1),y(:,2));
   hold off 
    
    
end


%%
%Task #3_7_1
% 1 - 924 x^2 + y^2 уст
% 2 - x*y не уст в 1й четверти
f_l = @(x,y) x.^2 + y.^2;
[xx, yy] = meshgrid(-0.5:0.1:3.5,0.1:0.1:5);
f_l_v = f_l(xx,yy);
hold on
contour(xx,yy,f_l_v);
for j = -0.5:0.2:0.5
    for k = 0.1:0.2:5
        y0 = [j,k];
        tspan = [0 1];
        [t, y] = ode45(@(t,y) odefcn_7_1(t,y),tspan,y0);
        n_q = numel(y(:,1));
        step = 8;
        plot(y(:,1),y(:,2),'red');
        n_q = fix(n_q/step)*step;  
        c_i = 0.01;
        for l = 1:step:n_q - step
            leng = ((y(l - 1 + step,1) - y(l,1)).^2 + (y(l - 1 + step,2) - y(l,2)).^2).^(1/2);
            if y(l,2) > 3 
                r = 0;
                b = 1;
                g = 0;
            else 
                if y(l,2) > 2
                r = 1;
                b = 1;
                g = 0;
                else
                    r = 0;
                    b = 1;
                    g = 0;
                end
            end    
            q = quiver(y(l,1),y(l - 1 + step,2),(y(l - 1 + step,1) - y(l,1))/leng,(y(l - 1 + step,2) - y(l,2))/leng,'Color',[r g b]); 
            %c_i = c_i + 0.2;
            %q.LineWidth = 1;
        end    
    end
end    
hold off
%%
%Task #3_7_2
f_l = @(x,y) x.*y;
[xx, yy] = meshgrid(-1:0.1:3.5,-0.5:0.1:5);
f_l_v = f_l(xx,yy);
hold on
contour(xx,yy,f_l_v);
n = 50;
x_m = linspace(0,2*pi,n);
for i = 1:n
        
        y0 = [sin(x_m(i)),cos(x_m(i))];
        tspan = [0.25, 0.81];
        [t, y] = ode45(@(t,y) odefcn_7_2(t,y),tspan,y0);
        n_q = numel(y(:,1));
        step = 10;
        plot(y(:,1),y(:,2),'r');
        n_q = fix(n_q/step)*step;    
        for l = 1:step:n_q - step
            leng = ((y(l - 1 + step,1) - y(l,1)).^2 + (y(l - 1 + step,2) - y(l,2)).^2).^(1/2);
            q = quiver(y(l,1),y(l - 1 + step,2),(y(l - 1 + step,1) - y(l,1))/leng,(y(l - 1 + step,2) - y(l,2))/leng,'Color','blue'); 
            %q.LineWidth = 1;
        end    
   
end    
hold off
%%
y0 = [0 1];
tspan = [0, 1/2];
[t, y] = ode45(@(t,y) odefcn_7_2(t,y),tspan,y0);
plot(t,y(:,1));
%%
%Task #3_8
xmesh = linspace(0,pi/2,5);
solinit = bvpinit(xmesh,@guess_8);

sol = bvp4c(@odefcn_8,@bcfcn_8,solinit);
hold on
plot (sol.x,sol.y(1,:), '-o');
x = linspace(0,pi/2,7);
y = 1 - cos(sol.x) - sin(sol.x);
d_2 = (trapz(sol.x,(sol.y(1,:) - y).^2)).^(1/2);
d_c = max(abs(y - sol.y(1,:)));
disp('||.||L_2:');
disp(d_2);
disp('||.||C:');
disp(d_c);
plot(sol.x,y);
hold off
disp(solinit);

%%
%Task #3_9
%f = @(x) sin(x);
%f_grad_x = cos(x);
f = @(x,y) x.^2 + y.^2 - 1;
f_grad_x = @(x,y) 2.*x;
f_grad_y = @(x,y) 2.*y;

[x_gr, y_gr] = meshgrid(-1:0.1:1,-1:0.1:1);
f_p = f(x_gr,y_gr);
eps = 0.1;
lambda = 0.1;
flag = 1;
l= 1;
n = 200;
x_m = linspace(-1,1,n);
y_m = linspace(-1,1,n);
f_m = f(x_m,y_m);
hold on
%plot3(x_m,y_m,f_m);

hold off

x_b = [1,1];
%x_b = 0;
%f_b_x = f_grad_x()
%f_b = 
f_b_x = f_grad_x(x_b(1),x_b(2));
f_b_y = f_grad_y(x_b(1),x_b(2));
f_b = [f_b_x,f_b_y];
while flag
    
    [x1, x2] = grd_fcn(x_b,f_b,lambda);
    l = l + 1;
    %disp(x1);
    %disp(x2);
    x_b = x2;
    f_b_x = f_grad_x(x_b(1),x_b(2));
    f_b_y = f_grad_y(x_b(1),x_b(2));
    f_b = [f_b_x,f_b_y];
    hold on
    contour(x_gr,y_gr,f_p,[f(x_b(1),x_b(2)),f(x_b(1),x_b(2))],'ShowText','on');
    plot3(x_b(1),x_b(2),f(x_b(1),x_b(2)),'r.','MarkerSize',20);
    
    hold off
    if l > 1000 
       flag = 0;
       disp('number of iteration - 100');
    end
    
    if norm(x2 - x1) <= eps
        
        flag = 0;
    end    
end

%%
%Task #3_9 одномерный случай 
f = @(x) x.^2;
f_grad_x = @(x) 2.*x;
l = 0;
x = linspace(-3,3,100);
hold on
plot(x,f(x));
hold off
x_b = 4;
f_b_x = f_grad_x(x_b);
lambda = 0.2;
eps = 0.0001;
flag = 1;
while flag
    
    [x1, x2] = grd_fcn(x_b,f_b_x,lambda);
    l = l + 1;
    
    x_b = x2;
    f_b_x = f_grad_x(x_b);
    
    
    hold on
    
    plot(x_b,f(x_b(1)),'r.','MarkerSize',20);
    hold off
    if l > 1000 
       flag = 0;
       disp('number of iteration - 100');
    end
    
    if ((x2 - x1).^2) <= eps
        
        flag = 0;
    end    
end
x = fminbnd(f,-2,2);
disp(x_b);
disp(x);
hold on
plot(x,f(x),'g.','MarkerSize',20);
%%
[t,y] = ode45(@(t,y) odefcn(t,y),[-1 1], [-2 2 5 3]);
hold on
plot(t,y(:,1),t,y(:,2), t,y(:,3),t,y(:,4));
plot(y(:,1),y(:,2));
hold off
axis equal;
%%

function dydt = odefcn(t,y)
dydt = zeros(4,1);
dydt(1) = y(3);
dydt(2) = y(4);
dydt(3) = 0;
dydt(4) = 0;
end

function dydt = odefcn_7_11(t,xy)
dydt = [(xy(2)-xy(1)+xy(1)*xy(2)); (xy(1)-xy(2)-xy(1)^2-xy(2)^3)];
end

function dydt = odefcn_5(t,y,alpha,beta,gamma,delta)
    dydt = zeros(2,1);
    dydt(1) = alpha*y(1) - gamma*y(1)*y(2);
    dydt(2) = -beta*y(2) + delta*y(1)*y(2);
end
function dydt = odefcn_6(t,y,a11,a12,a21,a22)
    dydt = zeros(2,1);
    dydt(1) = a11*y(1) + a12*y(2);
    dydt(2) = a21*y(1) + a22*y(2);

end
function theta = calAngleBetweenVectors(u, v)
dotUV = dot(u,v);
normU = norm(u);
normV = norm(v);

theta = acos(dotUV/(normU*normV));
end

function dydx = odefcn_8(x,y)
dydx = zeros(2,1);
dydx = [y(2) 1-y(1)]
end

function res = bcfcn_8(ya, yb)
    res = [ya(1) yb(1)];
end

function g = guess_8(x)
    g = [sin(x) cos(x)];
end


function dydt = odefcn_7_1(t,y)
    dydt = zeros(2,1);
    dydt(1) = y(2) - y(1) + y(1)*y(2);
    dydt(2) = y(1) - y(2) - y(1).^2 - y(2).^3;
end

function dydt = odefcn_7_2(t,y)
    dydt = zeros(2,1);
    dydt(1) = y(1).^2 + 2*y(2).^3;
    dydt(2) = y(1)*y(2).^2;
end

function [x1, x2] = grd_fcn(x,f_grad,lambda)
    x1 = x;
    x2 = x - lambda.*f_grad;    
end


