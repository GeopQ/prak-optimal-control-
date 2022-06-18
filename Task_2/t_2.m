%Задание № 2_1
f = @(x) 1./(1000000.*x);
n = 10;
a = -2;
b = 2;
x = linspace(a,b,n);
xx = linspace(a,b,2*n);

compareInterp(x,xx,f);

%%
%Задание № 2_2
%Пример 1
% nearest - x^2 - cubic(разность погрешностей) (высокая погрешность при большаой 1-й производной)
    g = @(x) x.^2;
    n = 50;
    x = linspace(0,5,n);
    xx = linspace(0,5,2*n);
    vn = interp1(x,g(x),xx,'nearest');
    vc = interp1(x,g(x),xx,'cubic'); 
    plot(xx,vn,xx,vc,xx,g(xx));
    
%%
%Пример 2
% cubic - 1/(1000000*x) (в ~5 раз)- linear(разница времени вычислений) (высокая погрешность при большаой 4-й производной)
    n = 1000;
    g = @(x) 1./(100000.*x);
    x = linspace(5,20,n);
    xx = linspace(5,20,2*n);
    t_l = 0;
    t_c = 0;
    for i = 1:500
        tic();
        vl = interp1(x,g(x),xx,'linear');
        t_l = t_l + toc();
        
        tic();
        vc = interp1(x,g(x),xx,'cubic'); 
        t_c = t_c + toc();
    end
    disp('Time nearest: ');
    disp(t_l/500);
    disp('Time cubic: ');
    disp(t_c/500);
    
    
%%
%Пример 3
%spline - необходимо минимум 4 точни для построения (высокая погрешность при большой 4-й производной)
    n = 3;
    g = @(x) sin(x);
    x = linspace(0,pi,n);
    xx = linspace(0,pi,2*n);
    %vl = interp1(x,g(x),xx,'linear');
    vs = interp1(x,g(x),xx,'spline');
    plot(xx,sin(xx),xx,vs);
    
    
%%
%Пример 4
%linear - разность погрешностей cubic и linear
    n = 1000;
    x = linspace (0,3,n);
    xx = linspace (0,3,3*n);
    g = @(x) sin(x.^3)+x.^3;
    v = g(x);
    vl = interp1(x,v,xx,"linear");
    vc = interp1(x,v,xx,"cubic");
    vv = g(xx);
    plot(xx,vv,xx,vl,xx,vc)
    disp('max err linear:');
    disp(abs(max(vv-vl)));
    disp('mar err cubic: ');
    disp(abs(max(vv-vc)));
    
%%
%Задание №2_3
a = 0;
b = 5;
%sin(10t) - высокая
%t - низкая
g = @(t) sin(10.*t);
gd = @(t) 10.*cos(10.*t);
n = 20;
ym = [];
for i = 1:n
   clear x;
   clear y;
   x = a + (i-1)*(b-a)/n:0.02:a + i*(b-a)/n; 
   ya = gd(x); 
   ym = [ym, (b-a)*max(ya)/n];
end
ym_m = max(abs(ym));
x_1 = linspace(a,b,n);
x_2 = linspace(a,b,2*n);
y_2 = g(x_2);
y_1 = g(x_1);
yq = interp1(x_1,y_1,x_2,'nearest');
yq = yq - y_2;
plot(x_1,ym_m*ones(numel(x_1)),x_1,abs(yq(2:2:end)),0,ym_m*1.2);
legend('apr','apost');
%%
%Задание №2_4
fn = @(x,n) 1*((x<= (1+mod(n,2.^(floor(log2(n)))))*2.^(-floor(log2(n)))) & (x>= mod(n,2.^(floor(log2(n))))*2.^(-floor(log2(n)))));
%fn = @(x,n) x.^n;
f = @(x) 0;
convergenceFunc(fn,f,0,1,50,'avsqr');
%подготовить примеры когда есть одна сходимость но нет другой
%fn = @(x,n) 1*((x<= (1+mod(n,2.^(floor(log2(n)))))*2.^(-floor(log2(n)))) & (x>= mod(n,2.^(floor(log2(n))))*2.^(-floor(log2(n)))));
%ИИС ср. кв. ^ avsqr [0,1]
%fn = x.^n [0,1/2] uncon
%интернет / равномерная ^

%%
%Задание №2_5
f = @(x) x.^2 - 10; 
           
a = -3;
b = 7;
n = 10;
fourierApprox(f,a,b,n,'lagger');
%%
%Задание №2_14
disp('___');
s = struct('ax',-0.2,'bx',0.2,'ay',-0.2,'by',0.2,'az',-0.2,'bz',0.2,'n',100,'fc','red','ec','green');
drawBall(2,s);
    

%%
%Задание №2_15

a = 1:10;
c = 'rbgrrrrrrr';
ed = 'None';

drawBalls(a,c,ed);
%%
K = wal(0,3);
disp(K);
%%
function result = compareInterp(x, xx, f)
    v = f(x);
    tic();
    %vq = interp1(x,v,xx,'nearest');
    %vq = interp1(x,v,xx,'linear');
    %vq = interp1(x,v,xx,'spline');
    vq = interp1(x,v,xx,'cubic');
    %disp(max(abs(vq(2:2:end)))); % для nearest
    %disp(max(abs(vq(1:2:end) - v)));
    disp(toc());
    plot(x,v,xx,vq);
    %legend('f(x)','m_nearest_f(x)');
    %legend('f(x)','m_linear');
    %legend('f(x)','m_spline');
    legend('f(x)','m_itrerp');
    %title('m_nearest');
end

function result = convergenceFunc(fn,f,a,b,n, convType)
    x = linspace(a,b,100);
    mov(1:n) = struct('cdata',[],'colormap',[]);
    v = f(x);
    v_i_draw = 0.*x;
    if numel(convType) ~= numel('pointwise')
        if convType == 'uncon'
           for i = 1:n
            v_i = fn(x,i);
            v_i_draw(i,:) = v_i;
            
            
           end
           maxc = max(max(abs(v_i_draw)));
           for i = 1:n
          
           plot(x,v_i_draw(i,:),x,v,0,maxc);
           str = string(convType) + ' | n = ' + string(i) + '| metric = '+ string(max(abs(v_i_draw(i,:) - v)));
           title(str)
           mov(i) = getframe();
           end 
        else
           for i = 1:n
            v_i = fn(x,i);
            v_i_draw(i,:) = v_i;
            
            
           end
           maxc = max(max(abs(v_i_draw)));
           for i = 1:n
          
           plot(x,v_i_draw(i,:),x,v,0,maxc);
           str = string(convType) + ' | n = ' + string(i) + '| metric = '+ string((trapz(x,(abs(v_i_draw(i,:) - v)).^2)).^(1/2));
           title(str)
           mov(i) = getframe();
           end 
        end
        
    else
        
        for i = 1:n
        v_i = fn(x,i);
        plot(x,v_i,x,v);
        mov(i) = getframe();       
        end
    end
    legend('fn(x)','f(x)');
end

function result = drawBall(alpha,s)
    [X,Y,Z] = meshgrid(s.ax:(s.bx - s.ax)/s.n:s.bx,s.ay:(s.by - s.ay)/s.n:s.by,s.az:(s.bz - s.az)/s.n:s.bz);  
     
    if alpha == inf 
        F = max(abs(X),abs(Y));
        F = max(F,abs(Z));
    else      
        f = @(x,y,z) abs(x).^(alpha)+abs(y).^(alpha)+abs(z).^(alpha);
        F = f(X,Y,Z);
    end
    gr = isosurface(X,Y,Z,F,1);
    if ~isempty(gr.faces)
        p = patch(isosurface(X,Y,Z,F,1));
        p.FaceColor = s.fc;
        p.EdgeColor = s.ec;
        camlight;
        view(3);
    else
        disp('variety is empty');
    end    
   
end

function result = fourierApprox(f,a,b,n,meth)
    mov(1:n) = struct('cdata',[],'colormap',[]);
   
    switch meth
        case 'ffourier'
             
            x = linspace(a,b,500);
            ff = f(x);
            x_tr = pi*(x-a)/(b-a);
            ff_v = zeros(1,500);
            ff_draw = zeros(n,500);
            mov(1:n) = struct('cdata', [],'colormap', []);
            for i = 1:n
                n_func = fourier(i);
                ff_n = n_func(x_tr);
                bi = trapz(x_tr,ff_n.*ff);
                ff_v = ff_v+2*bi.*ff_n/pi;
                ff_draw(i,:) = ff_v;
            end
   
            maxc = max(max(abs(ff_draw)));
            for i = 1:n
                plot(x,ff);
                hold on
                plot(0,maxc);
                plot(0,-maxc);
                plot(x,ff_draw(i,:));
                hold off
                mov(i) = getframe();
                
                
            end
            legend('f(x)','ffourier');

        
        case 'wal'
             
            L = b-a;
            xw = linspace(0,L,128);
            ff_vw = zeros(1,128);
            f_nw = f(xw);
            ff_draw = zeros(n,128);
            for i = 0:n-1
                
                ci = trapz(xw,f_nw.*wal(i,7))/L;
                ff_vw = ff_vw + ci*wal(i,7);
                ff_draw(i+1,:) = ff_vw;
      
            
            end
            maxc = max(max(abs(ff_draw)));
            
            for i = 1:n
                plot(xw,f_nw);
                hold on
                plot(0,maxc);
                plot(0,-maxc);
                plot(xw,ff_draw(i,:));
                hold off
                mov(i+1) = getframe();
                legend('f(x)','wal'); 
            end
            
        case 'lagger'
            L = b-a;
            xl = linspace(0,L,100);
            ff_vl = zeros(1,100);
            f_nl = f(xl);
            ff_draw = zeros(n,100);
            for i = 0:n-1
                plot(xl,f_nl);
                x_t = linspace(0,100,1000);
                Li = lagger(i);

                f_t = f(x_t);
                an = trapz(x_t,f_t.*Li(x_t).*exp(-x_t));
                an = an/trapz(x_t,Li(x_t).*Li(x_t).*exp(-x_t));
                ff_vl = ff_vl + an*Li(xl);
                ff_draw(i+1,:) = ff_vl;
                
                
            end    
            maxc = max(max(abs(ff_draw)));
            for i = 1:n
                plot(xl,f_nl);
                hold on
                plot(0,maxc);
                plot(0,-maxc);
                plot(xl,ff_draw(i,:));
                hold off
                mov(i) = getframe();
               legend('f(x)','lagger');
            end
            
            
     end
end
    
    
    
    
    

function result = drawBalls(alphas, colors, edges)
     a = 20;
     n = 100;
    [X,Y,Z] = meshgrid( -a:2*a/n:a,-a:2*a/n:a,-a:2*a/n:a);
    f = @(x,y,z,alpha) abs(x).^(alpha)+abs(y).^(alpha)+abs(z).^(alpha);
    F = f(X,Y,Z,alphas(1));
    gr = isosurface(X,Y,Z,F,1);
    if isempty(gr.faces)
       disp('variety is empty');
       exit
    end    
    hold on
    for i = 1:numel(alphas)
        clear F;
        F = f(X,Y,Z,alphas(i));
        p = patch(isosurface(X,Y,Z,F,1));
        p.FaceColor = colors(i);
        p.EdgeColor = edges;
        p.FaceAlpha = 0.5;
    end    
    camlight;
    view(3);
    hold off
end

function result = fourier(n)
    %fc = f.*cos(pi*i*x/L);
    %fs = f.*sin(pi*i*x/L);
   % ai = trapz(x,fc)/L;
    %bi = trapz(x,fs)/L;
    
    result =  @(x) sin(n*x);
    
end

%function result = wal(t,n,i)
 %   H = hadamard(n);
  %  flag = 1;
   % j = 1;
   % while (flag)
        
    %end    
    %id = fix(t*i -1e-9) + 1;
    
    %disp(H(i,id));
    %result = H(i,id);
%end

function result = lagger(n)
   
    result = @(x) laguerreL(n,x);
end


function y = wal(n,k)

if nargin ~= 2
    msg = ['The number of input arguments is not correct. Please',...
        'call the function using the format y = walshFunction2(n,k)'];
    error(msg)
elseif ~(isreal(k) && rem(k,1)==0)
    msg = 'Plsease insert an integer number for the parameter K';
    error(msg)
elseif ~(isreal(n) && rem(n,1)==0)
    msg = 'Plsease insert an integer number for the parameter N';
    error(msg)
    
elseif ~(n>=0 && n <2^k)
    msg = 'N must be an integer number between 0 and 2^K-1';
    error(msg)
end
N = 2^k;
if n == 0
    
    y = ones(1,N);
    
else
    j = 1:N;
    m = 1 + floor(log2(n));
    R = (-1).^floor(2^m.*((j-1)/N));
    y = R.*wal(2^m-1-n,k);
end

end
