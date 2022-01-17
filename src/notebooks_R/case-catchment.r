library(zoo)
library(sf)
library(hydromad)
library(rgeos)
library(tidyverse)

catchments=readLines(textConnection("SC #114
SC #106
SC #113
SC #107
SC #112
SC #108
SC #109
SC #157
SC #110
SC #111
SC #105
SC #161
"))

setwd("E:/cloudStor/source/projects/MW_BASE_RC8_411_4712/")
subc=read_sf("876a3af3-9508-4f42-802b-cf410c72404e.shp")
# cat_dem = raster("1.flt")

st_crs(subc)<-"+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs "
subc3<-st_transform(subc,"+proj=longlat +datum=WGS84 +no_defs ")

# With functional unit
x2=read_sf("91cf75b6-ce76-4d41-891f-2f1a2fe7c127.shp")
x2=subset(x2,IntSCs %in% catchments)
plot(x2[,"IntFUs"])
#st_area(x2) %>% sum

st_crs(x2)<-"+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs "
x2<-st_transform(x2,"+proj=longlat +datum=WGS84 +no_defs ")
write_sf(x2,"E:/cloudStor/PhDocs/Siting of the realtime Nitrate measurement/GISdata/mw_LU.shp")
# Up to here for Qian