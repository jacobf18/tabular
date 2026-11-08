arguments <- commandArgs(trailingOnly = TRUE)
setwd(arguments[1])
source("CANYONB.R")
measurements <- read.csv(arguments[2])
result <- CANYONB(date=measurements$date, lat=measurements$latitude,
                  lon=measurements$longitude, pres=measurements$pressure,
                  temp=measurements$temperature, psal=measurements$salinity,
                  doxy=measurements$oxygen, param="AT")
write.csv(data.frame(prediction=result$AT), arguments[3], row.names=FALSE)
