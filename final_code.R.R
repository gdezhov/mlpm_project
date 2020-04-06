# MLPM Project
# Sofia University “St. Kliment Ohridski”
# Faculty of Mathematics and Informatics
#
# Human activity recognition
# Author:  Georgi Dezhov
# Date  :  February 12, 2020




###Classify activities for a given person

#libraries
library(e1071)

#Choose person from 0 to 50
set.seed(1234)
x=sample(0:50,1)

#Load raw data for person x
#import raw data:

###A. accel_phone
A=c('data_160', x, '_accel_phone.txt')
f_name_accel_phone=paste(A, sep = '', collapse = '')
accel_phone=read.csv(f_name_accel_phone, header = F)
rm(f_name_accel_phone)
rm(A)
rm(x)

#Add columns for measure(accel=0 or gyro=1) and device(phone=0 or watch=1)
accel_phone=cbind(accel_phone, 
                  as.factor(rep(0, length(accel_phone$V1))),
                  as.factor(rep(0, length(accel_phone$V1))))


#Hand oriented activity column 'hand'=1; non-hand oriented activity 'hand'=0:
accel_phone[,9:10]=1

#add names of the columns
names(accel_phone)=c('Subject_id',
                     'Activity_code',
                     'Timestamp',
                     'X','Y','Z',
                     'measure',
                     'device', 
                     'hand',
                     'dist')


attach(accel_phone)
nh=which('accel_phone$Activity_code'== 'A'|
           accel_phone$Activity_code== 'B'|
           accel_phone$Activity_code== 'C'|
           accel_phone$Activity_code== 'D'|
           accel_phone$Activity_code== 'E'|
           accel_phone$Activity_code== 'M')
accel_phone[nh,9]=0
accel_phone$hand=as.factor(accel_phone$hand)
detach(accel_phone)
rm(nh)

# Z as.numeric
accel_phone$Z=as.numeric(accel_phone$Z)

# #Normalizes Numeric Data (X,Y,Z) To A Given a(0,1) Scale.
accel_phone$X=scale(accel_phone$X)
accel_phone$Y=scale(accel_phone$Y)
accel_phone$Z=scale(accel_phone$Z)

#add feature dist=sqrt(X^2+Y^2+Z^2)
accel_phone[,10]=round(sqrt(accel_phone[,4]^2+
                              accel_phone[,5]^2+
                              accel_phone[,6]^2), 4)

#check fo missing values
head(accel_phone)
sum(is.na(accel_phone))

#descriptive statistics
str(accel_phone)
summary(accel_phone)

#Sort by timestamp
accel_phone=accel_phone[order(accel_phone$Timestamp),]


####devide the frame by hand activity
A0_00=accel_phone[which(accel_phone$hand==0),]
head(A0_00)

A1_00=accel_phone[which(accel_phone$hand==1),]
head(A1_00)

###sliding window

#Sliding window step: 200=10sec/(50sec/1000)
sws=199

###for dist activity A0
dist_A0_00=matrix(0,nrow = (length(A0_00$dist))-sws,ncol = (sws+1))
for (i in 1:(length(A0_00$dist)-sws)) {
  dist_A0_00[i,]=t(A0_00[i:(i+sws),10])
}

dist_A0_00=as.data.frame(dist_A0_00)

#create data frame with feaatures
f_A0_00=data.frame(rep(0,length(A0_00$dist)-sws),
                   rep(0,length(A0_00$dist)-sws),
                   rep(0,length(A0_00$dist)-sws),
                   rowMeans(dist_A0_00),
                   apply(dist_A0_00,1,sd),
                   apply(dist_A0_00,1,skewness),
                   apply(dist_A0_00,1,kurtosis))
rm(dist_A0_00)
rm(A0_00)
rm(i)

#add names of the columns
names(f_A0_00)=c('hand',
                 'measure',
                 'device',
                 'means',
                 'st_devs',
                 'skews',
                 'kurts')

head(f_A0_00)
par(mfrow = c(2,2), mar= c(3, 4, 1, 1) + 0.1)
plot(f_A0_00$means, main='non hand means')
plot(f_A0_00$st_devs, main = 'non hand st_devs')
plot(f_A0_00$skews, main = 'non hand skews')
plot(f_A0_00$kurts, main = 'non hand kurts')

###for dist activity A1

dist_A1_00=matrix(0,nrow = (length(A1_00$dist))-sws,ncol = (sws+1))
for (i in 1:(length(A1_00$dist)-sws)) {
  dist_A1_00[i,]=t(A1_00[i:(i+sws),10])
}

dist_A1_00=as.data.frame(dist_A1_00)

#create data frame with feaatures
f_A1_00=data.frame(rep(1,length(A1_00$dist)-sws),
                   rep(0,length(A1_00$dist)-sws),
                   rep(0,length(A1_00$dist)-sws),
                   rowMeans(dist_A1_00),
                   apply(dist_A1_00,1,sd),
                   apply(dist_A1_00,1,skewness),
                   apply(dist_A1_00,1,kurtosis))

rm(dist_A1_00)
rm(A1_00)
rm(i)

#add names of the columns
names(f_A1_00)=c('hand',
                 'measure',
                 'device',
                 'means',
                 'st_devs',
                 'skews',
                 'kurts')

head(f_A1_00)
par(mfrow = c(2,2), mar= c(3, 4, 1, 1) + 0.1)
plot(f_A1_00$means, main = 'hand means')
plot(f_A1_00$st_devs, main = 'hand st_devs')
plot(f_A1_00$skews, main = 'hand skews')
plot(f_A1_00$kurts, main = 'hand kurts')

#split data frame into training, validation, and testing
####split f_A0_00
spec = c(train = .7, test = .3)
g = sample(cut(
  seq(nrow(f_A0_00)), 
  nrow(f_A0_00)*cumsum(c(0,spec)),
  labels = names(spec)
))

res_A0_00 = split(f_A0_00, g)

#To check the results:
sapply(res_A0_00, nrow)/nrow(f_A0_00)
addmargins(prop.table(table(g)))
rm(g)
rm(spec)

####split f_A1_00
spec = c(train = .7, test = .3)
g = sample(cut(
  seq(nrow(f_A1_00)), 
  nrow(f_A1_00)*cumsum(c(0,spec)),
  labels = names(spec)
))

res_A1_00 = split(f_A1_00, g)

#To check the results:
sapply(res_A1_00, nrow)/nrow(f_A1_00)
addmargins(prop.table(table(g)))
rm(g)
rm(spec)

###data frame to train
df_train=rbind(res_A0_00[[1]],res_A1_00[[1]])
df_train[,1]=as.factor(df_train[,1])

###data frame to test
df_test=rbind(res_A0_00[[2]],res_A1_00[[2]])


#clean
rm(res_A0_00)
rm(res_A1_00)

#### Logaritmic regresion
#data structure
str(df_train)

#model1
model1=glm(hand~means, family="binomial",data=df_train)
summary(model1)


#model2
model2=glm(hand~means+st_devs,data=df_train, family="binomial")
summary(model2)

#model3
model3=glm(hand~means+st_devs+skews, family="binomial",data=df_train)
summary(model3)

#model4
model4=glm(hand~means+st_devs+skews+kurts, family="binomial",data=df_train)
summary(model4)


#glm.fit
glm.fit = glm(hand~means+st_devs+skews+kurts, 
              data = df_train, 
              family = binomial)

coef(glm.fit)

#predict
glm.probs = predict(glm.fit,type = "response")
glm.probs[1:5]

glm.pred = ifelse(glm.probs > 0.5, '1', '0')

#confusion matrix
attach(df_train)
table(glm.pred, hand)
mean(glm.pred==hand )
detach(df_train)

#predict on new data
glm.probs.new =predict(glm.fit, df_test, type="response")
glm.probs.new[1:5]
glm.pred.new= ifelse(glm.probs.new > 0.5, '1', '0')

#confusion matrix
attach(df_test)
table(glm.pred.new, hand)
mean(glm.pred.new==hand )
detach(df_test)
