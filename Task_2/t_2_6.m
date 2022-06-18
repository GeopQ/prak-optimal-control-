
x = linspace(-16,20,1000);
f = @(x) sin(x);
%%
f_d = f(x);
[fmax, imax] = max(f_d);
TF = islocalmin(f_d);
hold on
plot(x,f_d,x(TF),f_d(TF),'g.',x(imax),fmax,'r.','MarkerSize',20);

x_max = x(imax)*ones(1,numel(x(TF)));
f_max = fmax*ones(1,numel(f_d(TF)));

v_distant_x = x_max - x(TF);
v_distant_y = f_max - f_d(TF);
v_distant_x = v_distant_x.^2;
v_distant_y = v_distant_y.^2;
v_distant = v_distant_y + v_distant_x;
v_distant = v_distant.^(1/2);
[n_min, imin] = min(v_distant(1,1:numel(f_d(TF))));
x_v = x(TF);
y_v = f_d(TF);
x_m_d = x_v(imin);%?????????????
y_m_d = y_v(imin);%?????????????
t = 0:0.01:1;


x_t = x(imax) + (x_m_d-x(imax))*t;
y_t = fmax + (y_m_d-fmax )*t;
comet(x_t,y_t);

