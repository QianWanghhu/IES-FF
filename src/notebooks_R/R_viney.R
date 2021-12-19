library(latticeExtra)
x =expand.grid(NSE=seq(0.3,0.6,length.out = 10),B=seq(-30/100,30/100,length.out=100)) 
x$viney = with(x,NSE-5*abs(log(1+B))^2.5)

thres = 0.5-5*abs(log(1-0.2))^2.5

lattice::contourplot(viney~NSE+B,data=x,at=c(0.25,0.3,round(thres, 3),0.35,0.4,0.45))+
  layer(panel.abline(h=0.20,v=0.5))

0.5-5*abs(log(1-0.2))^2.5

