disp(fix(30/4));
%%
%Задание №7
x = @(t) 2*sin(3*t + 2);
y = @(t) sin(t);
getEqual(x,y,1,10,50);
axis equal;

%%
%Задание №8

drawSet1(@kekw,30);
%@el
%@rhomb
%@square
%@ELLIPS_LEBESQUE
%@RHOMBUS_LEBESQUE
%@SQUARE_LEBESQUE


%%
%Задание №9
fun = @(x) x(1).^2 + x(2).^2 - 1;
opts = struct('l1',0,'l2',-1,'x0',[0,-1]);
%l1,l2 - направление
[val, x] = supportLebesque(@CIRCLE_LEBESQUE, opts); %  plot(x_t,y_t,'c');
disp(val);
disp(x);

%%
%Задание №10
drawPolar(@RHOMBUS_DRAWPOLAR_ZERO, 40);
%@SQUARE_DRAWPOLAR_ZERO  с 0
%@ELLIPS_DRAWPOLAR_ZERO_N без 0
%@SQUARE_DRAWPOLAR_ZERO_N  без 0
%@RHOMBUS_DRAWPOLAR_ZERO с 0
%@RHOMBUS_DRAWPOLAR_ZERO_N без 0
%%
function result = getEqual(f,g,t0,t1,N)
    t = linspace(t0,t1,3000);
    x = f(t);
    y = g(t);
    
    M = 1:N;
    M(N) = 3000;
    flag = 1;
    while flag
        max = 0;
        for i = 1:N-1
            if norm([x(M(i+1)) - x(M(i)),y(M(i+1)) - y(M(i))]) >= max
               max = norm([x(M(i+1)) - x(M(i)),y(M(i+1)) - y(M(i))]); 
               j = i;
            end    
            
        end
        if j == 1
            flag = 0;
        else
            M(j) = M(j) + 1;
        end
        
    end
    hold on
  
    dt = (t1- t0)/(N-1);
    
    T = zeros(1, N);
    for i = 1:N
        T(i) = t0 + (i-1)*dt;
    end
    
    plot(f(T),g(T),'k.','MarkerSize',20);
    plot(x,y,f(t(M)),g(t(M)),'r.','MarkerSize', 10);
    %plot(f(t(1)),g(t(1)),'g.','MarkerSize',30);
    %plot(f(t(3000)),g(t(3000)),'b.','MarkerSize',30);
    for i =1:length(M)
       text(f(t(M(i))),g(t(M(i))),num2str(i)); 
       %text(f(T(i)),g(T(i)),num2str(i),'Color','blue'); 
    end
    
    hold off
    r = zeros(2,N);
    r(1,:) = f(M);
    r(2,:) = g(M);
    normM = 0;
    normT = 0;
    for i = 1:N - 1
       % disp('____');
        %disp(i);
        %disp('dot1: ');
        %disp(f(t(M(i))));
        %disp(g(t(M(i))));
        %disp('dot2: ');
        %disp(f(t(M(i+1))));
        %disp(g(t(M(i+1))));
        %disp('distant: ');
        %disp(norm([f(t(M(i+1))) - f(t(M(i))),g(t(M(i+1))) - g(t(M(i)))])); 
        normM = normM + norm([f(t(M(i+1))) - f(t(M(i))),g(t(M(i+1))) - g(t(M(i)))]);
        normT = normT + norm([f(T(i+1)) - f(T(i)),g(T(i+1)) - g(T(i))]);
    end
    disp('result:');
    disp(normM/N);
    disp(normT/N);
    result = r;
end

function result = drawSet(rho, N)
    hold on
    t = linspace(0,2*pi - 2*pi./N ,N);
    y_1 = sin(t);
    y_2 = cos(t);
    v = [y_1(1),y_2(1)];
    [val, point] = rho(v,N);
    t_1 = point;
    t_first = point;
    for i = 2:N
    
    v = [y_1(i),y_2(i)];
    [val, point] = rho(v,N);
    
   
    x_p_e = linspace(t_1(1),point(1),50);
    y_p_e = linspace(t_1(2),point(2),50);
    plot(x_p_e,y_p_e,'b');
    
    
    c_apr_1 = -(t_1(1)*y_1(i-1)+t_1(2)*y_2(i-1));
    c_apr_2 = -(point(1)*y_1(i)+point(2)*y_2(i));
    x_apr = (c_apr_1*y_2(i)-c_apr_2*y_2(i-1))/(y_1(i)*y_2(i-1)-y_1(i-1)*y_2(i));
    y_apr = (c_apr_2*y_1(i-1)-c_apr_1*y_1(i))/(y_1(i)*y_2(i-1)-y_1(i-1)*y_2(i));     
    x_p_a_1 = linspace(t_1(1),x_apr,100);
    y_p_a_1 = linspace(t_1(2),y_apr,100);  
    x_p_a_2 = linspace(x_apr,point(1),100);
    y_p_a_2 = linspace(y_apr,point(2),100);
    plot(x_p_a_1,y_p_a_1,'r',x_p_a_2,y_p_a_2,'r');
    
    
    
    t_1 = point;
    end
    x_p_e = linspace(t_1(1),t_first(1),50);
    y_p_e = linspace(t_1(2),t_first(2),50);
   
    plot(x_p_e,y_p_e,'b');
    
    
    c_apr_1 = -(t_1(1)*y_1(N)+t_1(2)*y_2(N));
    c_apr_2 = -(t_first(1)*y_1(1)+t_first(2)*y_2(1));
    x_apr = (c_apr_1*y_2(1)-c_apr_2*y_2(N))/(y_1(1)*y_2(N)-y_1(N)*y_2(1));
    y_apr = (c_apr_2*y_1(N)-c_apr_1*y_1(1))/(y_1(1)*y_2(N)-y_1(N)*y_2(1));     
    x_p_a_1 = linspace(t_1(1),x_apr,100);
    y_p_a_1 = linspace(t_1(2),y_apr,100);  
    x_p_a_2 = linspace(x_apr,t_first(1),100);
    y_p_a_2 = linspace(y_apr,t_first(2),100);
    plot(x_p_a_1,y_p_a_1,'r',x_p_a_2,y_p_a_2,'r');
    
 
    hold off
end

function result = drawPolar(rho, N)
   
    t = linspace(0,2*pi-2*pi/N,N);
    y_1 = sin(t);
    y_2 = cos(t);
    v = [y_1(1),y_2(1)];
    [val, point] = rho(v);
    pointl = v./abs(val);
    point_1 = pointl;
    polar_x = zeros(1,N);
    polar_y = zeros(1,N);   
    polar_x(1) = pointl(1);
    polar_y(1) = pointl(2);
    hold on
    
    for i = 2:N
         v = [y_1(i),y_2(i)];
        [val, point] = rho(v);
        point_2 = v./abs(val);
        polar_x(i) = point_2(1);
        polar_y(i) = point_2(2);
        x_t = linspace(polar_x(i-1),polar_x(i),50);
        y_t = linspace(polar_y(i-1),polar_y(i),50); 
    end
    x_t = linspace(polar_x(N),polar_x(1),50);
    y_t = linspace(polar_y(N),polar_y(1),50); 
   
   
    K = convhull(polar_x,polar_y);
    plot(polar_x(K),polar_y(K), 'c');
    p = patch(polar_x(K),polar_y(K),'green');
    drawSet1(rho,20);
    axis equal;
    hold off
end

function [val, point] = supportLebesque(f, opts)
    A = [0,1];
    b = 0;
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    l1 = opts.l1;
    l2 = opts.l2;
    x0 = opts.x0;
    fun = @(x) -(l1 * x(1) + l2 * x(2));
    save params f;
    nonlcon = @unitdisk;
    [x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon);
    point = x;
    val = -fval;
end

function result = drawSet1(rho, N)
    hold on
    t = linspace(0,2*pi - 2*pi./N ,N);
    y_1 = sin(t);
    y_2 = cos(t);
    v = [y_1(1),y_2(1)];
    [val, point] = rho(v);
    t_1 = point;
    t_first = point;
    for i = 2:N
    
    v = [y_1(i),y_2(i)];
    [val, point] = rho(v);
    
   
    x_p_e = linspace(t_1(1),point(1),50);
    y_p_e = linspace(t_1(2),point(2),50);
    plot(x_p_e,y_p_e,'b');
    
    
    c_apr_1 = -(t_1(1)*y_1(i-1)+t_1(2)*y_2(i-1));
    c_apr_2 = -(point(1)*y_1(i)+point(2)*y_2(i));
    x_apr = (c_apr_1*y_2(i)-c_apr_2*y_2(i-1))/(y_1(i)*y_2(i-1)-y_1(i-1)*y_2(i));
    y_apr = (c_apr_2*y_1(i-1)-c_apr_1*y_1(i))/(y_1(i)*y_2(i-1)-y_1(i-1)*y_2(i));     
    x_p_a_1 = linspace(t_1(1),x_apr,100);
    y_p_a_1 = linspace(t_1(2),y_apr,100);  
    x_p_a_2 = linspace(x_apr,point(1),100);
    y_p_a_2 = linspace(y_apr,point(2),100);
    plot(x_p_a_1,y_p_a_1,'r',x_p_a_2,y_p_a_2,'r');
    
    
    
    t_1 = point;
    end
    x_p_e = linspace(t_1(1),t_first(1),50);
    y_p_e = linspace(t_1(2),t_first(2),50);
   
    plot(x_p_e,y_p_e,'b');
    
    
    c_apr_1 = -(t_1(1)*y_1(N)+t_1(2)*y_2(N));
    c_apr_2 = -(t_first(1)*y_1(1)+t_first(2)*y_2(1));
    x_apr = (c_apr_1*y_2(1)-c_apr_2*y_2(N))/(y_1(1)*y_2(N)-y_1(N)*y_2(1));
    y_apr = (c_apr_2*y_1(N)-c_apr_1*y_1(1))/(y_1(1)*y_2(N)-y_1(N)*y_2(1));     
    x_p_a_1 = linspace(t_1(1),x_apr,100);
    y_p_a_1 = linspace(t_1(2),y_apr,100);  
    x_p_a_2 = linspace(x_apr,t_first(1),100);
    y_p_a_2 = linspace(y_apr,t_first(2),100);
    plot(x_p_a_1,y_p_a_1,'r',x_p_a_2,y_p_a_2,'r');
    
 
    hold off
end





function [val, point] = SUPPORTDRAWPOLAR(f2, opts)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    l1 = opts.l1;
    l2 = opts.l2;
    x0 = opts.x0;
    fun = @(x) -(l1 * x(1) + l2 * x(2));
    save params f2;
    nonlcon = @unitdisk2;
    [x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon);
    point = x;
    val = -fval;
end

function [val, point] = CIRCLE_LEBESQUE(y)
   f = @(x) x(1).^2 + x(2).^2 - 1;
   opts = struct('l1',y(1),'l2',y(2),'x0',[0,-2]);
   [val, point] = supportLebesque(f,opts);
end

function [val, point] = SQUARE_LEBESQUE(y)
    f = @(x) 1 - max(abs(x(1)),abs(x(2) + 2));
    opts = struct('l1',y(1),'l2',y(2),'x0',[0,-2]);
    [val, point] = supportLebesque(f,opts);
end

function [val, point] = RHOMBUS_LEBESQUE(y)
    a = 0;
    b = 2;
    f = @(x) 1 - abs(x(1)+a) - abs(x(2) + b);
    opts = struct('l1',y(1),'l2',y(2),'x0',[0,-2]);
    [val, point] = supportLebesque(f,opts);
end

function [val, point] = RHOMBUS_LEBESQUE_AN(y)
    a = 0;
    b = 2;
    f = @(x) 1 - abs(x(1)+a) - abs(x(2) + b);
    t = linspace(-a - 1, a +1,100);
    for i = 1:100
        
    end
    opts = struct('l1',y(1),'l2',y(2),'x0',[0,-2]);
    [val, point] = supportLebesque(f,opts);
end

function [val, point] = ELLIPS_LEBESQUE(y)
    f = @(x) (x(1))^2-1+(x(2)+2)^2;
    opts = struct('l1',y(1),'l2',y(2),'x0',[3,-7/4]);
    [val, point] = supportLebesque(f,opts);
end
%_____________________________________________________________________
function [val, point] = el(l, N)
    x = 1;
    y = 1;
    a = 1/2;
    b = 1/10;
    C = ((l(1)*(1/a)).^2 + (l(2)*(1/b)).^2).^(1/2);
    val = dot([x,y],[l(1),l(2)]).^(1/2);
    
    point = [(l(1)*((1/a).^2))/C,(l(2)*((1/b).^2))/C] + [x,y];
   
end

function [val, point] = kekw(l)
    q1 = [1,1];
    q2 = [0,5];
    q3 = [6,5];
    r = 2;
    val = max(max(dot(l,q1),dot(l,q2)),dot(l,q3)) + abs(r);
    k = max(max(dot(l,q1),dot(l,q2)),dot(l,q3));
    q1_n = q1./(norm(q1));
    q2_n = q2./(norm(q2));
    q3_n = q3./(norm(q3));
    switch k
        case dot(l,q1)
            point = [q1(1),q1(2)] + q1_n.*r;
        case dot(l,q2)
            point = [q2(1),q2(2)] + q2_n.*r;
        case dot(l,q3)
            point = [q3(1),q3(2)] + q3_n.*r;
    end
end

function [val, point] = rhomb(l, N)
    x = 1;
    y = 3;
    T = 2;
    T1 = 5;
    val = dot([x,y],l) + T*max(abs(l(1)),abs(l(2)));
   
    if abs(l(1)*T) > abs(l(2)*T1) 
       if dot(l,[T,0]) > dot(l,[-T,0]) 
            point = [x,y] + [T,0];
       else
            point = [x,y] + [-T,0];
       end
    else
        if dot(l,[0,T1]) > dot(l,[0,-T1]) 
             point = [x,y] + [0,T1];
        else
             point = [x,y] + [0,-T1];
        end    
    end
    
end

function [val, point] = square(l, N)
    x = 1;
    y = 3;
    T = 2;
    val = dot(l,[x,y]) + T*(abs(l(1)) + abs(l(2))); 
    point = [x,y] + [T,T].*((l(1) >= 0) && (l(2) >= 0)) + [-T,T].*((l(1) < 0) && (l(2) > 0)) + [-T,-T].*((l(1) <= 0) && (l(2) <= 0)) + [T,-T].*((l(1) > 0) && (l(2) < 0));
   
    
    
    
    
    
end
%_____________________________________________________________________
function [val, point] = ELLIPS_DRAWPOLAR_ZERO_N(y)
    f = @(x) (2*x(1))^2-1+(4*x(2))^2;
    opts = struct('l1',y(1),'l2',y(2),'x0',[3,1/4]);
    [val, point] = SUPPORTDRAWPOLAR(f,opts);
end

function [val, point] = SQUARE_DRAWPOLAR_ZERO(y)
    f = @(x) 2 - max(abs(x(1)),abs(x(2)));
    opts = struct('l1',y(1),'l2',y(2),'x0',[0,0]);
    [val, point] = SUPPORTDRAWPOLAR(f,opts);
end

function [val, point] = SQUARE_DRAWPOLAR_ZERO_N(y)
    f = @(x) 1 - max(abs(x(1) - 2),abs(x(2)+2));
    opts = struct('l1',y(1),'l2',y(2),'x0',[2,-2]);
    [val, point] = SUPPORTDRAWPOLAR(f,opts);
end

function [val, point] = RHOMBUS_DRAWPOLAR_ZERO(y)
    f = @(x) 1 - abs(x(1)) - abs(x(2));
    opts = struct('l1',y(1),'l2',y(2),'x0',[0,0]);
    [val, point] = SUPPORTDRAWPOLAR(f,opts);
end

function [val, point] = RHOMBUS_DRAWPOLAR_ZERO_N(y)
    f = @(x) 1 - abs(x(1)-2) - abs(x(2));
    opts = struct('l1',y(1),'l2',y(2),'x0',[2,0]);
    [val, point] = SUPPORTDRAWPOLAR(f,opts);
end

function [c, ceq] = unitdisk(x)
    load params f
    ceq = f(x);
    c = [];
end

function [c, ceq] = unitdisk2(x)
    load params f2
    ceq = f2(x);
    c = [];
end

