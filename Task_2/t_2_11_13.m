%Задание №2_11 - 13
t = 1:10;
[x,y] = meshgrid(-5:0.25:5,-5:0.25:5);
z = @(x,y,a) a.*(sin(a*x) + a*cos(y));
%save('example.mat');
%%
nFrames = 10;
mov(1:nFrames) = struct('cdata',[],'colormap',[]);

for i = 1:nFrames
    z_value = z(x,y,t(i));
   
    %z = sin(x) + cos(y);
    TF_max = islocalmax(z_value,2);
    TF = islocalmin(z_value,2);
    M_max = TF_max.*z_value;
    M = TF.*z_value;
    TF_1_max = islocalmax(M_max);
    TF_1 = islocalmin(M);
   
    
    surf(x,y,z_value);
    
    hold on
    plot3(x.*TF_1,y.*TF_1,z_value.*TF_1,'r.','MarkerSize',40);
    plot3(x.*TF_1_max,y.*TF_1_max,z_value.*TF_1_max,'g.','MarkerSize',40);
    hold off
   
    
    
    mov(i) = getframe();
  
end
mov1 = VideoWriter('a.avi');
mov1.FrameRate = 1;
open(mov1);
writeVideo(mov1,mov);
close(mov1);
%save('example.mat','mov');
%%
load('example.mat');
%movie(mov);
%%
b = 2;
[x1,y1] = meshgrid(-2:0.2:2);
z_value1 = z(x1,y1,b);
contour(z_value1,1);
%%
mov1 = VideoWriter('a.avi');
open(mov1);
writeVideo(mov1,mov);
close(mov1);
%save('example.mat','mov');
%%


%Задание №2_13
m = [1,5,5;1,-10,5];
viewEaten(m,20,2);
%%


%%
function result = viewEaten(points, L, p)
    n = numel(points);
    max_c = max(max(abs(points),L));
    min_c = -max_c;
    [x,y] = meshgrid(min_c-1:0.1:max_c+1);
    F = x*0;
    for i = 1:n/2          
        G = ((points(1,i) - x).^p + (points(2,i) - y).^p).^(1/p);
        F = F + abs(G);
    end
    disp('Min L: ');
    disp(min(min(F)));
    disp('L: ');
    disp(L);
    
    hold on
    C = contour(x,y,F,[L,L]);
    fill(C(1,2:end),C(2,2:end),'green');
    plot(points(1,1:n/2),points(2,1:n/2),'r.','MarkerSize',10);
    hold off
end


%%
%1
alpha = 2;%2,3,4,5
t = 0.5;
%[X,Y,x_l,y_l] = reachset(alpha, t);
%hold on
%plot(X,Y,'Color','b','LineWidth',3);
%plot(x_l,y_l,'Color','r','LineWidth',3);
%xlabel('x_1');
%ylabel('x_2');
%hold off
t_1 = 0.3;
t_2 = 0.5;
filename = 'name.avi';
N = 10;
reachsetdyn(alpha,t_1,t_2,N,filename);

%%
%Функции-----------------------------------------

function [out1, out2, out3, out4] = reachset(alpha, t_1)
    t_0 = 0;
    tspan = [t_0 t_1];
    options = odeset('Events',@eventfcn);
    options_psi = odeset('Events',@eventfcn_psi);
    [t_plus, x_plus] = ode45(@(t_plus,x_plus) odefcn_s_plus(t_plus,x_plus,alpha), tspan, [0, 0],options);
    hold on
   % plot(x_plus(:,1),x_plus(:,2),'Color','black');
    hold off
    n = numel(x_plus(:,2));
    XM = [];
    YM = [];
    if (t_plus(n) <= t_1)
       tau = 0; 
    else
       tau = t_plus(n); 
    end    
    alph = alpha;
    x_T = [];
    y_T = [];
    for i=1:n-1
        b = 1;
        alpha = alph;
        x_1_b = x_plus(i,1);
        x_2_b = x_plus(i,2);
        tspan = [tau,t_1];
        if(tau == t_1)
            b = 0;
        end 
        W_plus_x = [];
        W_plus_y = [];
        W_minus_x = [];
        W_minus_y = [];
        while b
            alpha = alpha*(-1);
            j = -sign(alpha);
           
            [t_psi,x_psi] = ode45(@(t_psi,x_psi) odefcn_psi(t_psi,x_psi,alpha),tspan,[x_1_b x_2_b j 0], options_psi);
            n_t = numel(t_psi);
            t_psi_e = t_psi(n_t);
            hold on 
            XM = [XM, x_psi(:,1)'];
            YM = [YM, x_psi(:,2)'];
            %plot(x_psi(:,1),x_psi(:,2),'Color','black');
            hold off
            
            if t_psi_e == t_1
                b = 0;
                
                x_T = [x_T, x_psi(n_t,1)];
                y_T = [y_T, x_psi(n_t,2)];
            else
                if sign(alpha) < 0
                    disp('kek');
                    W_minus_x = [W_minus_x, x_psi(n_t,1)];
                    W_minus_y = [W_minus_y, x_psi(n_t,2)]; 
                else
                    disp('kek');
                    W_plus_x = [W_plus_x, x_psi(n_t,1)];
                    W_plus_y = [W_plus_y, x_psi(n_t,2)];
 
                end
                
                tspan = [t_psi_e, t_1];
                x_1_b = x_psi(n_t,1);
                x_2_b = x_psi(n_t,2);
            end
        end    
    end
    
    XM = [XM, x_T];
    YM = [YM, y_T];
    out1 = [XM,];
    hold on
    %plot(x_T,y_T,'Color','blue');
    
    hold off
%---------------------------------------------------------------------------------------------------------------------------------------------------------
    options_psi = odeset('Events',@eventfcn_psi);
    tspan = [t_0 t_1];
    [t_minus, x_minus] = ode45(@(t_minus,x_minus) odefcn_s_plus(t_minus,x_minus,alpha), tspan, [0, 0],options);
    hold on
    %plot(x_minus(:,1),x_minus(:,2),'Color','red');
    hold off
    n = numel(x_minus(:,2));
    if (t_minus(n) <= t_1)
       tau = 0; 
    else
       tau = t_minus(n); 
    end    
    x_T = [];
    y_T = [];
    for i=1:n-1
        b = 1;
        alpha = -alph;
        x_1_b = x_minus(i,1);
        x_2_b = x_minus(i,2);
        tspan = [tau,t_1];
        if(tau == t_1)
            b = 0;
        end
        W_plus_x = [];
        W_plus_y = [];
        W_minus_x = [];
        W_minus_y = [];
        while b
            alpha = alpha*(-1);
            j = -sign(alpha);
           
            [t_psi,x_psi] = ode45(@(t_psi,x_psi) odefcn_psi(t_psi,x_psi,alpha),tspan,[x_1_b x_2_b j 0], options_psi);
            n_t = numel(t_psi);
            t_psi_e = t_psi(n_t);
            XM = [XM, x_psi(:,1)'];
            YM = [YM, x_psi(:,2)'];
            hold on 
          %  plot(x_psi(:,1),x_psi(:,2),'Color','black');
            hold off
            
            if t_psi_e == t_1
                b = 0;
                
                x_T = [x_T, x_psi(n_t,1)];
                y_T = [y_T, x_psi(n_t,2)];
            else
                if sign(alpha) < 0
                    W_minus_x = [W_minus_x, x_psi(n_t,1)];
                    W_minus_y = [W_minus_y, x_psi(n_t,2)];
                    
                else
                    W_plus_x = [W_plus_x, x_psi(n_t,1)];
                    W_plus_y = [W_plus_y, x_psi(n_t,2)];
 
                end
                
                tspan = [t_psi_e, t_1];
                x_1_b = x_psi(n_t,1);
                x_2_b = x_psi(n_t,2);
            end
            
        end      
    end
    XM = [XM, x_T];
    YM = [YM, y_T];
    XM = XM';
    YM = YM';
    k = boundary(XM,YM,0.3);
    hold on
    %plot(XM(k),YM(k));
    %plot(x_T,y_T,'Color','green');
    hold off
    out1 = XM(k);
    out2 = YM(k);
    out3 = [fliplr(x_minus(:,1)),x_plus(:,1)];
    out4 = [fliplr(x_minus(:,2)),x_plus(:,2)];
end

function reachsetdyn(alpha, t1, t2, N, filename)
    mov(1:N) = struct('cdata',[],'colormap',[]);
    for i=1:N
        tau = t1 + ((t2 - t1).^(i))./N;
        [x, y, x1, y1] = reachset(alpha, tau);
        
        plot(x,y,'Color','b','LineWidth',3);
        
        mov(i) = getframe();
    end
    mov1 = VideoWriter('a.avi');
    mov1.FrameRate = 1;
    open(mov1);
    writeVideo(mov1,mov);
    close(mov1);
    
end

function dydt = odefcn_s_plus(t,y,alpha)
    dydt = zeros(2,1);
    dydt(1) = y(2);
    dydt(2) = alpha - y(1).^2 + 2.*sin(3.*y(1).^3) - y(1).*y(2);
end

function dydt = odefcn_psi(t,y,alpha)
    dydt = zeros(4,1);
    dydt(1) = y(2);
    dydt(2) = alpha - y(1).^2 + 2.*sin(3.*y(1).^3) - y(1).*y(2);
    dydt(3) = 2*y(4)*y(1) - 18*y(4)*((y(1))^(2))*cos(3*y(1)^3) + y(4)*y(2);
    dydt(4) = -y(3) + y(4)*y(1);
end

function [value, isterminal, direction] = eventfcn(t,x)
    value = x(2);
    isterminal = 1;
    direction = 0;
end

function [value, isterminal, direction] = eventfcn_psi(t,x)
    value = x(4);
    isterminal = 1;
    direction = 0;
end