%%
%Задание №1 ++++++
a = input('enter a ');
b = input('enter b ');
n = input('enter n ');
x = a : 1/n : b;
disp(x);

y = cos(x.^3 - 5*abs(x));
[ymin, imin] = min(y);
[ymax, imax] = max(y);
y_1 = [ymin, ymax];
x_1 = [x(imin), x(imax)];
 
plot(x,y,x_1,y_1,'r.','MarkerSize',20);
xlabel('x');
ylabel('y');

%%
%Задание №2 ++++++
n = input ('enter n '); 
p = primes(n);
%disp(p);
x = max(p);

if (n ~= x)
    disp('n not prime');
else
    disp('n is prime');
end    
    v = 7 : 14 : n;
    %disp(v);
   
    l = (1:n)';
    A = repmat(l,1,n);
    A = A + ones(n,n);
    disp(A);
    
    b = 1:(n+1)*(n+1);
    
    B = reshape(b,[n+1,n+1]);
    c = B(:);
    B = B.';
    disp(B);
    
    disp(c);
    D = B(:,n:n+1);
    disp(D);


%%
%Задание №3
B = -10 + 20*rand(5,8);
B = round(B);
d = diag(B);
disp(B);
max_d = max(d);
B1 = B.';
disp(B1);
c_v = prod(B1)./sum(B1);
disp(max(c_v));
disp(min(c_v));
    
B_sorted = sortrows(B,'descend');
disp(B_sorted);
%%
%Задание №4
n = input('enter n :');
m = input('enter m ( m mod 2 = 0 ) : ');
A = rand(n,m);
R = A(1:2:n,2:2:m);
B = A(2:2:n,1:2:m);
G(1:2:n,1:1:m/2) = A(1:2:n,1:2:m);
G(2:2:n,1:1:m/2) = A(2:2:n,2:2:m);
disp(A);
disp(R);    
disp(B);
disp(G);


%%
%Задание №5
n = input('enter n ');
m = input('enter m ');

x = rand(n,1);
y = rand(m,1);
disp(x);
disp(y);
C = repmat(y,n,1);
K = repmat(x,m,1);
K = sortrows(K);
C(1:m*n,2) = K;

disp(C);

%%
%Задание №6
n = 2;
B = rand(3,n); 
disp(B);
B1 = repmat(B,1,n);
B = B.';
B2 = repmat(B,1,n);
B2 = B2.';
b = B2(:);
B3 = reshape(b,[3,n*n]);
Rez_m = cross(B1,B3);
disp(Rez_m);
R(1:n*n) = ((Rez_m(1,1:n*n)).^2 + (Rez_m(2,1:n*n)).^2 +(Rez_m(3,1:n*n)).^2).^(1/2);
disp(numel(R));
C = reshape(R,n,n);
disp(C);
clear 
%%
%Задание №7 +++++
n = input('enter n ');
m = input('enter m ');
a = rand(1,n);
b = rand(1,m);
disp(a);
disp(b);
max_a = max(a);
min_a = min(a);
max_b = max(b);
min_b = min(b);
rez_1 = max_a - min_b;
rez_2 = max_b - min_a;
rez_v = [rez_1, -1*rez_1, rez_2, -1*rez_2];
disp(max(rez_v));

%%
%Задание №8
n = input('enter n ');
k = input('enter k ');
A = rand(k,n);
disp(A);
A1 = repmat(A,1,n);
A2 = repmat(A.',1,n);
A2 = A2.';
a = A2(:);
A3 = reshape(a,[k,n*n]);
A_rez = A1 - A3;
R(1:n*n) = (sqrt(sum(A_rez(1:k,1:n*n).^2)));
C = reshape(R,[n,n]);
disp(C);
%%
%Задание №9
n = input('enter n ');


t1_v = [];
t2_v = [];
for i = 1:n
    A = rand(i, i);
    B = rand(i, i);
    t_end_1 = zeros(20,1);
    t_end_2 = zeros(20,1);
    for j = 1:20
        tic
        C = A + B;
        t_end_1(j) = toc;
        clear C;
        tic
        C = matrix_sum(A,B);
        t_end_2(j) = toc;
        clear C;
    end
    t1 = median(t_end_1);
    t2 = median(t_end_2);
    clear t_end_1; 
    clear t_end_2;
    t1_v = [t1_v,t1 ]; %C = A + B
    t2_v = [t2_v,t2 ]; %C = matrix_sum(A,B)
end
hold on
plot(1:n, t1_v,'m-','MarkerSize',10);
plot(1:n, t2_v,'r-','MarkerSize',10)
legend('A + B','matrix_sum(A,B)');  
hold off




%%
%Задание №10
n = input('enter length ');
disp('enter the sequens of numbers ');
vec = [];
for i = 1:n
    x = input(':');
    vec = [vec x];
end    
if vec == fliplr(vec)
    disp('vector is simetrical');
else
    disp('vector is not simetrical');
end    
%%
%Задание №11
%unifrd \ unifcdf
n = input('enter n ');
a = input('enter a ');
b = input('enter b ');
R = a*rand(1,n);
disp(R);
R_t = R > b;

j = sum(R_t);  
disp(j);
disp(j/n);
if j/n > a/(2*b) 
    disp('n  j/n > a/(2*b)');
else
    disp('v j/n < a/(2*b)');
end
%%
%Задание №12

n = 200;
a = -5;
b = 5;
y_r = [];
y_s = [];

for i = a:10/n:b
   x_i = a:1/n:i;
   y_i = exp(-x_i.^2);
   y_r = [y_r, rectangles(x_i,y_i)];
   y_s = [y_s, simpson(x_i,y_i)];
end
x = a:1/n:b;
y = exp(-x.^2);
l = a:10/n:b;

%plot(x,y,l,y_r,l,y_s);
%legend('y = exp(-x^2)','Y_rec','Y_simp');
%__________________________________________________



v_t_t = [];%массив времени для trapz
v_s_t = [];%массив времени для simpson
v_r_t = [];%массив времени для rectangles
v_t = [];%массив разности для trapz
v_s = [];%массив разности для simpson
v_r = [];%массив разности для rectangles

for h = 0.01:0.01:1
    x = a:h:b;
    y = exp(-x.^2);
    x_t = a:h/2:b;
    y_t = exp(-x_t.^2);
   
    tic
    simpson(x,y);  
    v_s_t = [v_s_t, toc];
    
    tic
    rectangles(x,y);
    v_r_t = [v_r_t, toc];
    
    tic
    trapz(x,y);
    v_t_t = [v_t_t, toc];
    
    v_s = [v_s, abs(simpson(x,y)-simpson(x_t,y_t))];
    v_r = [v_r, abs(rectangles(x,y)-rectangles(x_t,y_t))];
    v_t = [v_t, abs(trapz(x,y)-trapz(x_t,y_t))];

end

h = 0.01:0.01:1;
%plot(h,v_s,h,v_r,h,v_t);
%legend('s_diff','r_diff','t_diff');


plot(h,v_s_t,h,v_r_t,h,v_t_t);
legend('simp_t','rect_t','trapz_t');

%loglog(h,v_r);
%loglog(h,v_t);
%loglog(h,v_s);
%loglog(h,t_r);
%loglog(h,t_t);
%loglog(h,t_s);
%legend('rectangles_dif','trapz_dif','simpson_dif','r_time','t_time','s_time');
%xlabel('h');
%ylabel('err');

%%
%Задание №13
%Производная в точке x = 1
y = logspace(-10,1);
%y = cos(x);
%y_d = -sin(x);
g_c = y;
g_r = y;
g_c = abs(-sin(1) - (cos(1 + y) - cos(1 - y))./(2*y));
g_r = abs(-sin(1) - (cos(1 + y) - cos(1))./y);    
loglog(y,g_c,y,g_r);
legend('abs(y_d(1) - y_c(1))','abs(y_d(1) - y_r(1))');    
xlabel('x');
ylabel('y');
%%
function rez = rectangles(x,y)   
    n = numel(x);
    if (n >= 2) 
       % h = x(2) - x(1);
        rez = (x(2) - x(1))*sum(y(1:n-1));    
    else
        rez = 0;
    end
end
function rez = simpson(x,y)
    n = numel(x);
    if (n >= 3) 
        h = x(2) - x(1);
        rez = (h/3)*(4*sum(y(3:2:n-1)) + 2*sum(y(2:2:n-1)) + y(1) + y(n));
    else
        rez = 0;
    end
end    
function sum_rez = matrix_sum(A, B)
    C = ones(size(A));
    for i = 1:size(A(1,:))
        for j = 1:size(A(:,1))
            C(i,j) = A(i,j) + B(i,j);
        end    
    end
    sum_rez = C;
end
