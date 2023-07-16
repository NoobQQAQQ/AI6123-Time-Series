library(quantmod) #getSymbols
library(zoo)
library(xts)
library(tseries) # adf.test
library(TSA) # kurtosis, skewness
library(forecast)
library(ggplot2)
library(PerformanceAnalytics)

### ----- Get data from Yahoo ----- ###
stockname <- 'AAPL'
stock.data <- getSymbols(stockname, from='2002-02-01', to='2017-02-01', src='yahoo', auto.assign = F) 
stock.data <- na.omit(stock.data) # remove possible na values
chartSeries(stock.data, theme = "white", name = stockname)
aapl.c <- stock.data[,4] # extract close price
names(aapl.c) <- 'Apple Stock Closing Prices (2002.02-2017.02)'

### ----- Examine data ----- ###
# plot closing price
global.xlab <- 'Date (yyyy-mm-dd)'
global.ylab <- 'Closing Price ($)'
ggplot(aapl.c, aes(time(aapl.c), as.matrix(aapl.c))) + geom_line(colour = 'red') +
  ylab(global.ylab) +
  xlab(global.xlab) +
  ggtitle(names(aapl.c))
# test stationarity
acfPlot(aapl.c)
pacfPlot(aapl.c)
adf.test(aapl.c)
kpss.test(aapl.c, null = "Trend")
kpss.test(aapl.c, null = "Level")
pp.test(aapl.c)

### ----- Data transformation ----- ###
# transform to log-returns
global.ylab <- 'Log-returns*100'
aapl.logr <- diff(log(aapl.c))*100
aapl.logr <- na.omit(aapl.logr) # remove possible na values
names(aapl.logr) <- 'Log-returns*100 on AAPL Closing Prices (2002.02-2017.02)'
ggplot(aapl.logr, aes(time(aapl.logr), as.matrix(aapl.logr))) + geom_line(colour = 'red') +
  ylab(global.ylab) +
  xlab(global.xlab) +
  ggtitle(names(aapl.logr))
# test stationarity
acfPlot(aapl.logr)
pacfPlot(aapl.logr)
adf.test(aapl.logr)
kpss.test(aapl.logr, null = "Trend")
kpss.test(aapl.logr, null = "Level")
pp.test(aapl.logr)

# Plot timeplot and ACF PACF Plots of Abs and Sqrt
acfPlot(abs(aapl.logr))
pacfPlot(abs(aapl.logr))
acfPlot(aapl.logr^2)
pacfPlot(aapl.logr^2)

# QQ Plot
qqnorm(aapl.logr)
qqline(aapl.logr)
skewness(aapl.logr) # -0.190054
kurtosis(aapl.logr) # 5.435621
chart.Histogram(aapl.logr, methods = c("add.density"),
                colorset=c("gray","red"), main="Histogram of Log-returns*100")

### ----- Split train/test set ----- ###
num_of_train <- (length(aapl.logr) - 30) %>% print()
aapl.logr.train <- head(aapl.logr, num_of_train)
aapl.logr.test <- tail(aapl.logr, round(length(aapl.logr) - num_of_train))

### ----- EACF ----- ###
eacf(aapl.logr.train)
eacf(aapl.logr.train^2)
eacf(abs(aapl.logr.train))

### ----- Fitted Models ----- ###
arma.04 = arma(aapl.logr.train, order=c(0,4))
summary(arma.04)
garch.11=garch(aapl.logr.train, order=c(1,1))
summary(garch.11)
AIC(garch.11)
garch.10=garch(aapl.logr.train, order=c(0,1))
summary(garch.10)
AIC(garch.10)

garch.22=garch(aapl.logr.train, order=c(2,2))
summary(garch.22)
AIC(garch.22)
garch.21=garch(aapl.logr.train, order=c(1,2))
summary(garch.21)
AIC(garch.21)
garch.20=garch(aapl.logr.train, order=c(0,2))
summary(garch.20)
AIC(garch.20)


garch.33=garch(aapl.logr.train, order=c(3,3))
summary(garch.33)
AIC(garch.33)
garch.32=garch(aapl.logr.train, order=c(2,3))
summary(garch.32)
AIC(garch.32)
garch.31=garch(aapl.logr.train, order=c(1,3))
summary(garch.31)
AIC(garch.31)
garch.30=garch(aapl.logr.train, order=c(0,3))
summary(garch.30)
AIC(garch.30)

### ----- GARCH Diagnostic Checking ----- ###
plot(residuals(garch.11),type='h',ylab='Standardized Residuals', main='GARCH(1,1)')
qqnorm(residuals(garch.11)); qqline(residuals(garch.11))
acfPlot(residuals(garch.11)^2, na.action=na.omit)
gBox(garch.11,method='squared') # above p-value

### ----- forecasting ----- ###
fit.garch11 <- garchFit(formula = ~garch(1, 1), 
                        data = aapl.logr.train, trace = F, cond.dist = "std")
pred.garch11 <- predict(fit.garch11, n.ahead = 30, plot=TRUE)
accuracy(pred.garch11$meanForecast, aapl.logr.test)

fit.garch22 <- garchFit(formula = ~garch(2, 2), 
                        data = aapl.logr.train, trace = F, cond.dist = "std")
pred.garch22 <- predict(fit.garch22, n.ahead = 30, plot=TRUE)
accuracy(pred.garch22$meanForecast, aapl.logr.test)

fit.garch33 <- garchFit(formula = ~garch(3, 3), 
                        data = aapl.logr.train, trace = F, cond.dist = "std")
pred.garch33 <- predict(fit.garch33, n.ahead = 30, plot=TRUE)
accuracy(pred.garch33$meanForecast, aapl.logr.test)

Box.test(residuals(fit.garch11), type="Ljung-Box")
plot(fit.garch11, which=3)  #Series with 2 Conditional SD Superimposed
plot(fit.garch11, which=13)  #QQ Plot
pred.garch11 <- predict(fit.garch11, n.ahead = 30, plot=TRUE)
accuracy(pred.garch11$meanForecast, aapl.logr.test)


