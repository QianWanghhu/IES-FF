library(latticeExtra)
x =expand.grid(NSE=seq(0.3,0.5,length.out = 10),B=seq(0,20/100,length.out=10)) 
x$viney = with(x,NSE-5*abs(log(1+B))^2.5)

thres = 0.5-5*abs(log(1+0.15))^2.5

lattice::contourplot(viney~NSE+B,data=x,at=c(0.25,0.3,thres,0.35,0.4,0.45))+
  layer(panel.abline(h=0.15,v=0.5))
