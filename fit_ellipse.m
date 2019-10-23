function ellipse = fit_ellipse(XY)
scale = max(max(XY));
XY = XY ./ scale;
centroid = mean(XY);   % the centroid of the data set
D1 = [(XY(:,1)-centroid(1)).^2, (XY(:,1)-centroid(1)).*(XY(:,2)-centroid(2)),...
      (XY(:,2)-centroid(2)).^2];
D2 = [XY(:,1)-centroid(1), XY(:,2)-centroid(2), ones(size(XY,1),1)];
S1 = D1'*D1;
S2 = D1'*D2;
S3 = D2'*D2;
T = -inv(S3)*S2';
M = S1 + S2*T;
M = [M(3,:)./2; -M(2,:); M(1,:)./2];
[evec,eval] = eig(M);
cond = 4*evec(1,:).*evec(3,:)-evec(2,:).^2;
A1 = evec(:,find(cond>0));
A = [A1; T*A1];
A4 = A(4)-2*A(1)*centroid(1)-A(2)*centroid(2);
A5 = A(5)-2*A(3)*centroid(2)-A(2)*centroid(1);
A6 = A(6)+A(1)*centroid(1)^2+A(3)*centroid(2)^2+...
     A(2)*centroid(1)*centroid(2)-A(4)*centroid(1)-A(5)*centroid(2);
A(4) = A4;  A(5) = A5;  A(6) = A6;
para = A/norm(A);

if para(2)^2-4*para(1)*para(3)<0
    ellipse.tilt = atan(-para(2)/(para(1)-para(3))) / 2;
    ct = cos(ellipse.tilt);
    st = sin(ellipse.tilt);
    a = para(1)*ct^2-para(2)*ct*st+para(3)*st^2;
    c = para(1)*st^2+para(2)*ct*st+para(3)*ct^2;
    d = para(4)*ct-para(5)*st;
    e = para(4)*st+para(5)*ct;
    f = para(6);

    ellipse.xc = -d/(2*a);
    ellipse.yc = -e/(2*c);
    cc = d^2/(4*a)+e^2/(4*c)-f;
    ellipse.xa = sqrt(cc/a);
    ellipse.ya = sqrt(cc/c);
    ellipse.general = para;
    
else
    ellipse = [];
end

end