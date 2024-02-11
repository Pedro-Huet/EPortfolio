library(zoo)
library(ggplot2)
library(tibble)
library(tidyr)
library(purrr)
library(dplyr)
library(stringr)
library(forecast)
library(xts)
library(tidyverse)
library(lubridate)
library(tseries)
library(astsa)
library(foreign)
library(timsac)
library(vars)
library(mFilter)
library(dynlm)
library(TTR)
library(fpp2)
library(bimets)
library(TSstudio)
library(lmtest)

#Definicion de Serie

ts(TCa, start = 2010, frequency = 12)
st1<-ts(TCa, start = 2010, frequency = 12)
st1
class(st1)
plot(st1)
start (st1);end(st1); frequency (st1)

#1 Observacion del comportamiento de la Serie

acf(st1)
pacf(st1)
adf.test(st1, alternative="stationary")

ndiffs(st1)

#2 Modelo MA

stMA<-arima(st1, order= c(0,0,1))
stMA
summary(stMA)
tsdiag(stMA)
residuals(stMA)
checkresiduals(stMA)
Box.test(residuals(stMA))

stMA2<-arima(st1, order= c(0,0,2))
stMA2
summary(stMA2)
tsdiag(stMA2)
residuals(stMA2)
checkresiduals(stMA2)
Box.test(residuals(stMA2))

stMA3<-arima(st1, order= c(0,0,3))
stMA3
summary(stMA3)
tsdiag(stMA3)
residuals(stMA3)
checkresiduals(stMA3)
Box.test(residuals(stMA3))

#Pronostico del Modelo.

forecast(stMA2, h = 12)
plot(forecast(stMA2, h = 12))

#3 Modelo ARIMA

stARIMA<-Arima(st1, order= c(1,1,2))
stARIMA
summary(stARIMA)
tsdiag(stARIMA)
residuals(stARIMA)
checkresiduals(stARIMA)
Box.test(residuals(stARIMA), type ="Ljung-Box")
adf.test(residuals(stARIMA))

forecast(stARIMA, h = 12)
plot(forecast(stARIMA, h = 12))
