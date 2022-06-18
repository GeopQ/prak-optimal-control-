%(t.^2).*(exp(-3*abs(t))) --> -18*(p.^2 - 3)./(p.^2 + 9).^3
%(t.^3 + 2*t)./(t.^4 + 4*t.^2 + 4) --> -i*(pi/2).^(1/2).*exp(-abs(p)*(2).^(1/2))
%exp(-3*t.^4).*cos(t)
%(abs(t) <= 1)*exp(-t.^2)


clc
clear
plotFT(figure,@func4,[],0.01,[-100,100],[ 0,20]);
%0.01 10 \ 0.001 100
%x = linspace(-1,1,100);


%%

function result_str = plotFT(hFigure, fHandle, fFThandle, step, inpLimVec, outLimVec)
    plotINFO = get(hFigure,'UserData');
    if(~isempty(plotINFO))
       if (isempty(outLimVec))
            outLimVec = plotINFO.outLimVec;
       end    
    end
    if isempty(outLimVec)
       outLimVec = [0 20]; 
    end
    figure(hFigure);
    a = inpLimVec(1);
    b = inpLimVec(2);
    c = outLimVec(1);
    d = outLimVec(2);
    T = abs(b - a);
    N = round(T/step);
    
    step_1 = T/N; %новый шаг дискретизации

    t=linspace(0.0000000001,b - a,N);
    f = zeros(1,N);
    sm = a/(b - a);
    c = ceil(sm);
    i = 1;
    while t(i)+c*(b-a)<=b
        f(i) = fHandle(t(i)+c*(b - a));
        i =i+1; 
        
    end
    add = step_1 - (-b+t(i)+c*(b - a));

    j = 0;
    while i+j<=N
        f(i+j) = fHandle(t(j+1)-step_1+inpLimVec(1)+add);%!
        j =j+1;
    end


   
    
    Y = T*fft(f)/(N);
    Y = [Y,Y];
    %disp(numel(Y));
    %disp(fft(f_res));
    d_f = 2*pi/T;
    v_Y = -N:1:N-1 ;
   
  %  disp(length(real(Y)));
  % disp(real(numel(fFThandle(v_Y))));
    
    v_Y = d_f.*v_Y;
    disp(length(v_Y)); 
    if (isempty(fFThandle))
        subplot(2,1,1);
        Pre = plot(v_Y,real(Y));
        xlabel('L');
        ylabel('re(L)');
        legend('apr. FT Re');
        
        axis([outLimVec(1) outLimVec(2) min(real(Y)) max(real(Y))]);
        subplot(2,1,2);
        
        Pim = plot(v_Y,imag(Y));
        axis([outLimVec(1) outLimVec(2) min(imag(Y)) max(imag(Y))]);
        xlabel('L');
        ylabel('im(L)');
        legend('apr. FT Im');
    else
        subplot(2,1,1);
        Pre = plot(v_Y,real(Y),v_Y,real(fFThandle(v_Y)));
        %Pre = plot(v_Y,real(Y));
        axis ([outLimVec(1) outLimVec(2) min(min(real(Y)),min(real(fFThandle(v_Y)))) max(max(real(Y)),max(real(fFThandle(v_Y))))]);
        xlabel('L');
        ylabel('re(L)');
        legend('FT Re','apr. FT Re');
        subplot(2,1,2);
        %disp(length(v_Y));
        %disp(fFThandle(v_Y));
        cf = 1;
        Pim = plot(v_Y,imag(Y)./cf,v_Y,imag(fFThandle(v_Y)));
        axis ([outLimVec(1) outLimVec(2) min(min(imag(Y)),min(imag(fFThandle(v_Y)))) max(max(imag(Y)),max(imag(fFThandle(v_Y))))]);
        xlabel('L');
        ylabel('im(L)');
        legend('F Im','apr. FT Im');
    end
    str = struct('Re_hangle',@Pre,'Im_hangle',@Pim,'outLimVec',outLimVec);
    
    if (isempty(plotINFO))
        plotINFO = str;
    else
        plotINFO(end) = str;
    end
    set(hFigure,'UserData',plotINFO);
    result_str = struct('nPoints',N,'step',step_1,'inpLimVec',[a,b],'outLimVec',[outLimVec(1), outLimVec(2)])
    
end



function f = func1(t)
    f = (t.^2).*(exp(-3*abs(t)));
end

function ft = ftfunc1(p)
    ft = -36*(p.^2 - 3)./(p.^2 + 9).^3;
    
end

function f = func2(t)
    f = t/(t^2 + 2);
end

function ft = ftfunc2(p)
    ft = 0 - sign(p)*i*(pi).*exp(-abs(p)*(2).^(1/2));%*i
end

function f = func3(t)
    f = exp(-3*t.^4).*cos(t)
end

function f = func4(t)
    f = (abs(t) <= 1).*exp(-t.^2)
end