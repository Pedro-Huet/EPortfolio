#Tarea Final

#Para poder usar las librerias es necesario invocarlas
library("normtest")
library("datasets")
library("Ecdat")
library("graphics")
library("lmtest")
library("stats")
library("dplyr")
library("car")
library("sem")
library("strucchange")
library("sandwich")

#1.Cargar base en cuestión

clase2mu<-read.csv("C:/Users/Eventos/Documents/Diplomado Econometría/Entregas/Tarea 5/FPN1.csv", header=T)
attach(clase2mu)

#Análisis de un modelo que explica el nivel de producto actual a partir de una función Cobb-Douglas (capital y trabajo) de tipo lineal corregida y con rezago (año anterior).

#2. Creación de las variables corregidas del modelo, 

lPIB<-log(PIB); lPIB
lStK<-log(StK); lStK
lL<-log(L); lL

#3. Rezagar las variables corregidas del modelo, 

laglStK<-lag(lStK); laglStK
laglL<-lag(lL); laglL
laglPIB<-lag(lPIB); laglPIB

#4. Modelo en cuestión:

modfin<-lm(lPIB~laglStK+laglL)
modfinalt1<-lm(lPIB~laglStK)
modfinalt2<-lm(lPIB~laglL)
summary(modfin)
plot(modfin)
valajust<-fitted(modfin)

#5. Visualización de residuos

resid<-residuals(modfin)
boxplot(resid)

#6. Pruebas de Normalidad de los resiudos.

#a) Gráfica Normal.

plot(modfin)

#b) Diagrama de caja y bigote.

boxplot(resid)

#c) Histograma.

hist(resid, freq=F, main="Residuales del modelo")
rug(jitter(resid))
lines(density(resid), col="red", lwd=3)

#d). Prueba Jarque-Bera.

jb.norm.test(resid, nrepl=2000)

#e). Prueba Shapiro-Will.

shapiro.test(resid)

#7. Pruebas de Autocorrelación.

#a) Gráfica de dispersión contra residuales.

plot(modfin)

residlag<-lag(resid)

plot(resid,residlag)
abline(lm(resid~residlag))

#b) Correlograma.

acf(resid, ci=0.95, xlab="Residuos del modelo", ylab="Autocorrelaciones", main="Correlograma del modelo", plot=TRUE)

#c) Correlograma parcial.

pacf(resid, ci=0.95, xlab="Residuos del modelo", ylab="Autocorrelaciones", main="Correlograma del modelo", plot=TRUE)

#d) Prueba Durbin-Watson.

dwtest(modfin)

#e) Breusch-Godfrey.

bgtest(modfin)

#8. Pruebas de Homocedasticidad.

#a) Gráfica de dispersión contra residuales.

plot(modfin)

#b) Prueba de Breusch-Pagan-Godfrey.

bptest(modfin)

#C) Prueba de White.

bptest(modfin, ~ laglStK*laglL + I(laglStK^2) + I (laglL), data = clase2mu)

#9. Pruebas de Multicolinealidad. 

#a) Correlaciones entre variables. 

cor(clase2mu)

#b) Regla de Klein.

mod1aux<-lm(laglStK~laglL)
mod2aux<-lm(laglL~laglStK)

summary(mod1aux)
summary(mod2aux)

#c) Regla de Theli.

reglatheil=0.03722-(0.03722-0.03622)-(0.03722-0.0808)-(0.03722-0.0817)

#d) Factor de Inflación de la Varianza.

vif(modfin) 

#10. Estimación correcta del modelo.

#a) Prueba RESET de Ramsay.

resettest(modfin, power=2:2)


#b) Prueba Log-likehood.

logLik(modfin)
logLik(modfinalt1)
logLik(modfinalt2)

#c) Prueba Akaike.

AIC(modfin)
AIC(modfinalt1)
AIC(modfinalt2)


#11. Cambios estructurales.

#a) Gráfico de Distribución de Observaicones

plot.ts(modfin)

#b) Prueba Chow

sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=5)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=15)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=25)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=35)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=45)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=55)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=65)
sctest(lPIB~laglStK+laglL, data=clase2mu, type="Chow", point=75)

#c) Prueba Cusum

prueba.cusum = efp(lPIB~laglStK+laglL, type = "OLS-CUSUM")
plot(prueba.cusum)