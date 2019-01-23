function [h1,h2] = conepd2a(BW,range,colorequal,rangeequal)
% Draw cone structure of 2 x 2 p.d. matrix

if nargin == 0;
    BW = 1;
    range = 1.5;
    colorequal = 'green';
    rangeequal = range;
end
if nargin == 1;
    range = 1.5;
    colorequal = 'green';
    rangeequal = range;
end
if nargin == 2; 
    colorequal = 'green';
    rangeequal = range;
end
if nargin == 3;  
    rangeequal = range;
end

% Create cone
[t,z] = meshgrid((0:0.05:2)*pi, (0:0.1:range));
a = z.*(1 + cos(t));
b = z.*(1 - cos(t));
c = z*sqrt(2).*sin(t);

% figure 
% set(gcf, 'Position', [0 0 600 600]) % [0 350 800 800]
if BW,
   h1 = surf(a,b,c,'FaceColor',0.9*[1 1 1],'EdgeColor',0.4*[1 1 1]);
else
   h1 = surf(a,b,c,'FaceColor','red','EdgeColor',[0.2 0.2 0.2]);
end
camlight; lighting phong
axis equal, axis([0 2.35 0 2.35 -1.2 1.2])
alpha(0.7), view([135 35])

% Add axes, labels
% line([0 2.35],[0 0],[0 0],'Color','black','LineWidth',2)
% line([0 0],[0 2.35],[0 0],'Color','black','LineWidth',2)
% line([0 0],[0 0],[-1.2 1.2],'Color','black','LineWidth',2)

h2 = line([0 rangeequal],[0 rangeequal],[0 0],'Color',colorequal,'LineWidth',2);

%  h = text(f*f*Lxy,0,0,'a'); set(h, 'Fontsize', 18)
%  h = text(0,f*f*Lxy,0,'b'); set(h, 'Fontsize', 18)
%  h = text(0,0,f*f*Lz/2,'c'); set(h, 'Fontsize', 18)

axis equal;

title('The cone of Sym^+(2)'); 