a = [1 2];
a = [a,a];
disp(a);
%%
%Task #3_7_1'

hold on
n = 10;
xx = linspace(-5,5);
yy = linspace(-5,5);
[x_1, y_1] = meshgrid(xx,yy);
f_L = @(x,y) x.^2 + y.^2;
f_L_d  = f_L(x_1,y_1);
u = y_1 - x_1 +x_1.*y_1;
v = x_1 - y_1 - x_1.^2 - y_1.^3;
q = quiver(x_1,y_1,u,v);
x_p = linspace(0,2*pi,n);
r = 5;
axis equal;
for i = 1:n
y0 = r.*[cos(x_p(i)),sin(x_p(i))];
tspan = [-10 10];
[t, y] = ode45(@(t,y) odefcn_7_11(t,y),tspan,y0);
for j = 1:numel(y(:,1)) - 1
    l_c = plot(y(j:j+1,1),y(j:j+1,2));
    l_c.Color = [1 j/numel(y(:,1)) 0];
end    




end
contour(x_1,y_1,f_L_d);
hold off
%%
%Task #3_7_2'
hold on
n = 20;
xx = linspace(-2,2);
yy = linspace(-2,2);
[x_1, y_1] = meshgrid(xx,yy);
f_L = @(x,y) x.*y;
f_L_d  = f_L(x_1,y_1);
u = x_1.^2 + 2*y_1.^3;
v = x_1.*y_1.^2;
q = quiver(x_1,y_1,u,v);
x_p = linspace(0,2*pi,n);
r = 1/2;
axis([-2 2 -2 2]);
for i = 1:n
    options = odeset('Events',@EventsFcn);
    y0 = r.*[cos(x_p(i)),sin(x_p(i))];
    tspan = [0 30];
    [t, y] = ode45(@(t,y) odefcn_7_2(t,y),tspan,y0,options);
for j = 1:numel(y(:,1)) - 1
    l_c = plot(y(j:j+1,1),y(j:j+1,2));
    l_c.Color = [1 j/numel(y(:,1)) 0];
end    
contour(x_1,y_1,f_L_d);
end
%%
function dydt = odefcn_7_11(t,xy)
dydt = [(xy(2)-xy(1)+xy(1)*xy(2)); (xy(1)-xy(2)-xy(1)^2-xy(2)^3)];
end

function dydt = odefcn_7_2(t,y)
    dydt = [y(1).^2 + 2*y(2).^3; y(1).*y(2).^2;];
end
   
function [value, isterminal, direction] = EventsFcn(t,y)   
    value = abs(y(1)).^2 + abs(y(2)).^2 - 4;
    isterminal = 1;
    direction = 0;  
end