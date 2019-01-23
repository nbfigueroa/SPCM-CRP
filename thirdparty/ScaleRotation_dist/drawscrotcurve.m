function drawscrotcurve(X,M,neval,rowid,iannotate,colorbydominantaxis);
% DRAWSCROTCURVE visualize a minimal SCAROT curve. Works for p = 2 and 3. 
% drawscrotcurve(X,Y) plots a sequence of ellipses or ellipsoids, that
% represent SCAROT curve from X to Y. 
%
% June 9, 2015 Sungkyu Jung.

if nargin == 2;
    neval = 7;
    rowid = [1 1];
    iannotate = true;
    colorbydominantaxis = false;
end
if nargin == 3;
    rowid = [1 1];
    iannotate = true;
    colorbydominantaxis = false;
end

if nargin == 4;
    iannotate = true;
    colorbydominantaxis = false;
end

if nargin == 5; 
    colorbydominantaxis = false;
end
[p,~]=size(X);
if  p  == 3
    
    [~, paramsscrot]=MSRcurve(M,X);
    [~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
    axisdisplaylength = 1.2*sqrt(max([diag(X);diag(M)]));
    rotaxis = real([paramsscrot.A(3,2), paramsscrot.A(1,3), paramsscrot.A(2,1)]);
    angle = norm(rotaxis);
    rotaxis = axisdisplaylength*rotaxis/angle; 
    % scaling = diag(paramsscrot.L)';
    axisdisplaylength = axisdisplaylength/1.5;
    coll = [1 0 0; 0 1 0; 0 0 1];
    for t = 1:neval
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
        unow = Uarray{t};
        dnow = diag(Darray{t});
        plotellipsoid(real(unow),dnow,colorbydominantaxis);
        hold on;
        for k = 1:3;
            plot3([0, sqrt(dnow(k))*unow(1,k)],...
                [0, sqrt(dnow(k))*unow(2,k)],...
                [0, sqrt(dnow(k))*unow(3,k)],...
                'linewidth',1,'color',coll(k,:));
        end
        plot3([-rotaxis(1), rotaxis(1)],...
            [-rotaxis(2), rotaxis(2)],...
            [-rotaxis(3), rotaxis(3)],...
            'k','linewidth',1,'color',[0 0 0]);
        xlim([-axisdisplaylength, axisdisplaylength] )
        ylim([-axisdisplaylength, axisdisplaylength] )
        zlim([-axisdisplaylength, axisdisplaylength] )
        axis off
    end
    if iannotate
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ 1);title('M');
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ neval);title('X');
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ floor((neval+1)/2));title(['MSR curve with distance: '...
            num2str(dist) ', angle: ' num2str(round(angle*180/pi))]);
    end
    
else
    
    
    
    [~, paramsscrot]=MSRcurve(M,X);
    [T,~,~,~,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
    axisdisplaylength = 1.2*sqrt(max([diag(X);diag(M)])); 
    for t = 1:neval
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
        plotellipse(matd(T(:,t)),[0 0 0]);
        xlim([-axisdisplaylength, axisdisplaylength])
        ylim([-axisdisplaylength, axisdisplaylength])
        axis equal;
        axis off; 
    end
    if iannotate
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ 1);title('M');
        subplot(rowid(1),neval, neval*(rowid(2)-1)+ neval);title('X');
    end
end


