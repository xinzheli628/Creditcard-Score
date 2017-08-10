setwd("C:/Users/lidianshuang/Desktop/R")

#### 一、数据预处理 ####
df1<- read.csv("51_data.csv",sep = ",")
head(df1)
df1<-df1[,-3]
colnames(df1)[1]<-"consumer_id"
colnames(df1)[3]<-"overday"
colnames(df1)[4]<-"borrow_state"
df1<-df1[,-5]
head(df1)

df2<-read.csv("fengkong_data.csv")
colnames(df2)[1]<-"consumer_id"
head(df2)

#设置匹配索引
index<-df2$consumer_id %in% df1$consumer_id
df2<-df2[index,]

index2<-df1$consumer_id %in% df2$consumer_id
df1<-df1[index2,]

dim(df1);dim(df2)

#合并数据
mydf<-merge(df1,df2,by="consumer_id")
head(mydf);dim(mydf)
#write.csv(mydf,"mydf.csv")

#去除无效列
mydf<-mydf[,c(-2,-4,-6)]
data<-mydf
accepts<-data
accepts<- accepts[,-18]
dim(accepts)
accepts$default <- ifelse(accepts$overday>45,1,0)
head(accepts)
table(accepts$default) ##数据不平衡，  Y: 1:0 = 10:1 需进行欠抽样平衡数据
inx<- table(accepts$default)

##1.1数据抽样##
bad<- accepts[accepts$default==1,]
good<- accepts[accepts$default==0,]
dim(bad);dim(good)
set.seed(1234)
select<-sample(1:nrow(good),length(good$consumer_id)*0.1)
good1=good[select,]
bad<- accepts[accepts$default==1,]
dim(bad);dim(good1)
colnames(bad);colnames(good1) ##3:1 

#确认列名称一致后，合并抽样后数据
sample<-rbind(bad,good1)
head(sample);dim(sample)
table(sample$default)

sample<-sample[,c(-1,-2)]

sample<-sample[sample$call_times>0 & sample$mean_time>0,]
table(sample$default);dim(sample)
sample<-na.omit(sample)
##入网时长为0的，将0替换为缺失值，然后进行插补
sample$online_time[sample$online_time==0] <- NA
colnames(sample)

write.csv(sample,"acc.csv")

##第一步准备数据集：把解释变量和被解释变量分开，不能用Y去预测X，并应用于模型
sample_x<-sample[,c("age.y","call_time","nocall_rate","night_rate","online_time","call_times","mean_time","linkman_cnt","long_rate","zm_score","hulu_score","first_black","org_cnt","sec_black","search_cnt","risk_score","risk_cnt")]
sample_default<-sample[,c("default")]

reject_x<-reject_x[,c("age.y","call_time","nocall_rate","night_rate","online_time","call_times","mean_time","linkman_cnt","long_rate","zm_score","hulu_score","first_black","org_cnt","sec_black","search_cnt","risk_score","risk_cnt")]


##对accepts数据进行插补
library(mice)
imp<-mice(sample_x,met="cart",m=1)#该方法只对数值变量进行插补，分类变量的缺失值保留
sample_x<-complete(imp)

##对reject数据进行插补
imp<-mice(rejects_x,met="cart",m=1)#该方法只对数值变量进行插补，分类变量的缺失值保留
rejects_x<-complete(imp)



########二、BP神经网络违约推断########

memory.limit(102400)
setwd("C:/Users/lidianshuang/Desktop/R")
library(AMORE)#载入包
set.seed(1234)#随机数设置
samp.rate=0.7#设置训练集比例

p2pdata<-read.csv("acc.csv")#导入数据
head(p2pdata);dim(p2pdata)
p2pdata<- na.omit(p2pdata)
str(p2pdata)#查看数据类型
#将整数型数据转为数值型
p2pdata$age.y=as.numeric(p2pdata$age)
p2pdata$call_time=as.numeric(p2pdata$call_time)
p2pdata$nocall_rate=as.numeric(p2pdata$nocall_rate)
p2pdata$night_rate=as.numeric(p2pdata$night_rate)
p2pdata$online_time=as.numeric(p2pdata$online_time)
p2pdata$call_times=as.numeric(p2pdata$call_times)
p2pdata$mean_time=as.numeric(p2pdata$mean_time)
p2pdata$linkman_cnt=as.numeric(p2pdata$linkman_cnt)
p2pdata$long_rate=as.numeric(p2pdata$long_rate)
p2pdata$zm_score=as.numeric(p2pdata$zm_score)
p2pdata$hulu_score=as.numeric(p2pdata$hulu_score)
p2pdata$first_black=as.numeric(p2pdata$first_black)
p2pdata$org_cnt=as.numeric(p2pdata$org_cnt)
p2pdata$sec_black=as.numeric(p2pdata$sec_black)
p2pdata$search_cnt=as.numeric(p2pdata$search_cnt)
p2pdata$risk_score=as.numeric(p2pdata$risk_score)
p2pdata$risk_cnt=as.numeric(p2pdata$risk_cnt)
p2pdata$default=as.numeric(p2pdata$default)

str(p2pdata)#查看数据类型

#数据归一化（数据压缩到0与1之间）
mean.vec=apply(p2pdata[,-18],2,mean)
sd.vec=apply(p2pdata[,-18],2,sd)
#range.vec=max.vec-min.vec
std=p2pdata[,-18]
for(i in 1:ncol(std))
{
  std[,i]=(std[,i]-mean.vec[i])/sd.vec[i]
}
#得到归一化以后的数据clean.data
clean.data=p2pdata
clean.data[,-18]=std
#clean.data<-clean.data[clean.data[,-21]<=3 & clean.data[,-21]>=-3,]
clean.data<-na.omit(clean.data)
#查看归一化效果
apply(clean.data,2,max)

#抽样
samp.index=sample(1:nrow(clean.data),size=floor(samp.rate*nrow(clean.data)))
#生成训练集与测试集
train=clean.data[samp.index,]
test=clean.data[-samp.index,]

##构造模型
net=newff(n.neurons = c(10,10,1),learning.rate.global = 0.001,momentum.global = 0.001,
          error.criterium = "LMS", hidden.layer = "tansig",
          output.layer = "purelin", method="ADAPTgdwm")
model=train(net,train[,-18],train[,18],error.criterium = "LMS",
            report=T,show.step=100,n.show=10)
test.predict=sim(model$net,test[,-18])
test.class=ifelse(test.predict>0.5,1,0)
library(gmodels)
CrossTable(test[,18], test.class, prop.chisq = FALSE, prop.c = F, 
           prop.r = F, dnn = c("actual", "predicted "))
sum(diag(table(test.class,test[,18])))/nrow(test)



##构建循环参数
arg1<-c(1:10)
arg2<-seq(from=1,to=10,by=1)
arg3<-seq(from=0.5,to=0.85,by=0.05)
varg1<-rep(arg1,each=(length(arg2)*length(arg3)))
varg2<-rep(rep(arg2,times=length(arg1)),each=length(arg3))
varg3<-rep(arg3,times=(length(arg1)*length(arg2)))
param<-cbind(varg1,varg2,varg3)
param<-as.data.frame(param)
result<-c()
library(gmodels)

##开始循环
for(i in 1:nrow(param))
{
  net=newff(n.neurons = c(28,param$varg1[i],param$varg2[i],1),learning.rate.global = 0.001,momentum.global = 0.001,
            error.criterium = "LMS", hidden.layer = "tansig",
            output.layer = "purelin", method="ADAPTgdwm")
  model=train(net,train[,-18],train[,18],error.criterium = "LMS",
              report=F,show.step=100,n.show=10)
  test.predict=sim(model$net,test[,-18])
  test.class=ifelse(test.predict>0.5,1,0)
  CrossTable(test[,18], test.class, prop.chisq = FALSE, prop.c = F, 
             prop.r = F, dnn = c("actual", "predicted "))
  precision<-sum(diag(table(test.class,test[,18])))/nrow(test)
  result<-rbind(result,cbind(param[i,],precision))
  i=paste(i/10,"%",sep="")
  print(i)
}


library("ROCR")##载入包
pred <- prediction(test.predict, test[,29])
perf <- performance(pred,'tpr','fpr')
plot(perf,colorize=TRUE)  #画图
auc <- performance(pred, measure = "auc")  #计算AUC
auc@y.values[[1]]





#### 三、用KNN算法进行违约推断 ####
setwd("C:/Users/lidianshuang/Desktop/R")
#1、导入数据和数据清洗
accepts<-read.csv("acc.csv")
rejects<-read.csv("rejects.csv")

names(rejects)
library(mice)
default<-accepts[,c("default")]
x<-accepts[,c("age.y","call_time","nocall_rate","night_rate","online_time","call_times","mean_time","linkman_cnt","long_rate","zm_score","hulu_score","first_black","org_cnt","sec_black","search_cnt","risk_score","risk_cnt")]
imp<-mice(x,met="cart",m=1)#该方法只对数值变量进行插补，分类变量的缺失值保留
x_imp<-complete(imp)

#标准化数据
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
x_imp<- as.data.frame(lapply(x_imp, normalize))

data<-cbind(default,x_imp)
data$default<-as.factor(data$default)
summary(data)
#构建训练集和测试集
set.seed(110)
select<-sample(1:nrow(data),length(data$default)*0.7)
train=data[select,-1]
test=data[-select,-1]
train.default=data[select,1]
test.default=data[-select,1]
#使用KNN算法
library(class)
result<-c()
for(i in 1:100){
  default_hat<-knn(train = train,test = test,cl=train.default,k=i)
  accuracy.knn<-sum(default_hat==test.default)/length(test.default)
  accuracy.knn
  result<-rbind(result,cbind(i,accuracy.knn))
  print(i)
}

#结果展现
require(gmodels)
CrossTable(x =test.default, y = default_hat,
           prop.chisq=FALSE)
agreement_KNN<- default_hat==test.default
table(agreement_KNN)

#!!运用于违约推断
setwd("C:/Users/lidianshuang/Desktop/R")
accepts<-read.csv("acc.csv")
rejects<-read.csv("rejects.csv")
##第一步准备数据集：把解释变量和被解释变量分开，这是KNN这个函数的要求
accepts_x<-accepts[,c("","","","")]
accepts_bad_ind<-accepts[,c("default")]
rejects_x<-rejects[,c("","","","")]
##第二步，进行缺失值填补和标准化，这也是KNN的要求
library(mice)
imp<-mice(accepts_x,met="cart",m=1)#该方法只对数值变量进行插补，分类变量的缺失值保留
accepts_x<-complete(imp)
imp<-mice(rejects_x,met="cart",m=1)#该方法只对数值变量进行插补，分类变量的缺失值保留
rejects_x<-complete(imp)

accepts_x<- as.data.frame(lapply(accepts_x, normalize))
rejects_x<- as.data.frame(lapply(rejects_x, normalize))

#第三步，建模并预测
library(class)
default<-knn(train = accepts_x,test = rejects_x,cl=accepts_default,k=30)
rejects<-cbind(default,rejects)
table(rejects$default)

#第四步，讲审核通过的申请者和未通过申请者进行合并
#不要忘记，accepts的数据是针对于违约用户的过度抽样
#因此，rejects也要进行同样比例的抽样


table(accepts$default)
rate<-table(accepts$default)[2]/table(accepts$default)[1]

set.seed(110)
rejects_bad<-rejects[rejects$default==1,]
rejects_good<-rejects[rejects$default==0,]
selected_good<-sample(1:nrow(rejects_good),nrow(rejects_bad)*(1/rate))
rejects_good=rejects_good[selected_good,]

rejects_sample<-rbind(rejects_bad,rejects_good)
summary(rejects_sample$default)

accepts<-read.csv("accepts.csv")

names(accepts)
names(rejects_sample)
library(sqldf)
accepts_1<-sqldf("select application_id,bad_ind,vehicle_year,vehicle_make,
                 bankruptcy_ind,tot_derog,tot_tr,age_oldest_tr,tot_open_tr,tot_rev_tr,
                 tot_rev_debt,tot_rev_line,rev_util,fico_score,purch_price,msrp,down_pyt,
                 loan_term,loan_amt,ltv,tot_income,veh_mileage,used_ind 
                 from accepts")
rejects_sample_1<-sqldf("select application_id,bad_ind,vehicle_year,vehicle_make,
                        bankruptcy_ind,tot_derog,tot_tr,age_oldest_tr,tot_open_tr,tot_rev_tr,
                        tot_rev_debt,tot_rev_line,rev_util,fico_score,purch_price,msrp,down_pyt,
                        loan_term,loan_amt,ltv,tot_income,veh_mileage,used_ind 
                        from rejects_sample")
data<-rbind(accepts_1,rejects_sample_1)
write.csv(data,file="data.csv")




#### 四、Logistic回归 ####
#计算相关系数矩阵
corre<-cor(sample)#org_cnt & mutil_plat & search_cnt 强相关
corre

#随机抽样，建立训练集与测试集
set.seed(1234)
select<-sample(1:nrow(accepts),length(accepts$consumer_id)*0.7)
train=accepts[select,]
test=accepts[-select,]
attach(train)
plot(zm_score,default)

#单变量分析
lg<-glm(default ~zm_score,family=binomial(link='logit')) ## *** 显著
summary(lg)

lg2<-glm(default ~age.y,family=binomial(link='logit')) ## 不显著
summary(lg2)

lg3<-glm(default ~call_time,family=binomial(link='logit')) ## * 次显著
summary(lg3)

lg4<-glm(default ~nocall_rate,family=binomial(link='logit')) ## *** 显著
summary(lg4)

lg5<-glm(default ~night_rate,family=binomial(link='logit')) ## *** 显著
summary(lg5)

lg6<-glm(default ~online_time,family=binomial(link='logit')) ## *** 显著
summary(lg6)

lg7<-glm(default ~call_times,family=binomial(link='logit')) ## 不显著
summary(lg7)

lg8<-glm(default ~mean_time,family=binomial(link='logit')) ## 不显著
summary(lg8)

lg9<-glm(default ~linkman_cnt,family=binomial(link='logit')) ## *** 显著
summary(lg9)

lg10<-glm(default ~long_rate,family=binomial(link='logit')) ## 不显著
summary(lg10)

lg11<-glm(default ~hulu_score,family=binomial(link='logit')) ## ** 显著
summary(lg11)

lg12<-glm(default ~first_black,family=binomial(link='logit')) ## ** 显著
summary(lg12)

lg13<-glm(default ~org_cnt,family=binomial(link='logit')) ## 不显著
summary(lg13)

lg14<-glm(default ~sec_black,family=binomial(link='logit')) ## ** 显著
summary(lg14)

lg15<-glm(default ~search_cnt,family=binomial(link='logit')) ## 不显著
summary(lg15)

lg16<-glm(default ~mutil_plat,family=binomial(link='logit')) ## 不显著
summary(lg16)

lg17<-glm(default ~risk_score,family=binomial(link='logit')) ## 不显著
summary(lg17)

lg18<-glm(default ~risk_cnt,family=binomial(link='logit')) ## 不显著
summary(lg18)

lg<-glm(default ~zm_score+age.y+call_time+nocall_rate+night_rate+online_time+call_times+mean_time+linkman_cnt+long_rate+hulu_score+first_black+org_cnt+sec_black+search_cnt+mutil_plat+risk_score+risk_cnt,family=binomial(link='logit'))
summary(lg)

#逐步回归
lg_ms<-step(lg,direction = "both")
summary(lg_ms)

##p值转换
train$lg_p<-predict(lg_ms, train) 
summary(train$lg_p)
train$p<-1/(1+exp(-1*train$lg_p))
summary(train$p)
write.csv(test,"testset.csv")
##检测多重共线性
library(car)
vif(lg)

#模型评估
test$lg_p<-predict(lg_ms, test) 
test$p<-(1/(1+exp(-1*test$lg_p)))
summary(test$p)
test$out<-ifelse(test$p<0.5,0,1)
table(test$default,test$out)

#计算准确率
rate2<-sum(test$out==test$default)/length(test$default)
rate2


library(ROCR)
pred_Te <- prediction(test$p, test$default)
perf_Te <- performance(pred_Te,"tpr","fpr")
pred_Tr <- prediction(train$p, train$default)
perf_Tr <- performance(pred_Tr,"tpr","fpr")
plot(perf_Te, col='blue',lty=1);
plot(perf_Tr, col='black',lty=2,add=TRUE);
abline(0,1,lty=2,col='red')

#绘制ROC曲线
lr_m_auc<-round(as.numeric(performance(pred_Tr,'auc')@y.values),3)
lr_m_str<-paste("Mode_Train-AUC:",lr_m_auc,sep="")
legend(0.3,0.4,c(lr_m_str),2:8)

lr_m_auc<-round(as.numeric(performance(pred_Te,'auc')@y.values),3)
lr_m_ste<-paste("Mode_Test-AUC:",lr_m_auc,sep="")
legend(0.3,0.2,c(lr_m_ste),2:8)

library(pROC)
plot.roc(default~p,train,col="1")->r1
rocobjtr<- roc(train$default, train$p)
auc(rocobjtr)
lines.roc(default~p,test,col='2')->r2
rocobjte <- roc(test$default, test$p)
auc(rocobjte)
roc.test(r1,r2)

#绘制洛伦兹曲线
pred_Tr <- prediction(train$p, train$default)
tpr <- performance(pred_Tr,measure='tpr')@y.values[[1]]
depth <- performance(pred_Tr,measure='rpp')@y.values[[1]]
plot(depth,tpr,type='l',main='Lorenz图',ylab='查全率(tpr)',xlab='深度(depth)')

#绘制累积提升度曲线
library(ROCR)
pred_Tr <- prediction(train$p, train$default)
lift <- performance(pred_Tr,measure='lift')@y.values[[1]]
depth <- performance(pred_Tr,measure='rpp')@y.values[[1]]
plot(depth,lift,type='l',main='lift图',ylab='lift',xlab='depth')

#绘制K-S曲线
pred_Tr <- prediction(train$p, train$default)
tpr <- performance(pred_Tr,measure='tpr')@y.values[[1]]
fpr <- performance(pred_Tr,measure='fpr')@y.values[[1]]
ks<-(tpr-fpr)
depth <- performance(pred_Tr,measure='rpp')@y.values[[1]]
plot(depth,ks,type='l',main='K-S曲线',ylab='KS值',xlab='深度(depth)')
kslable<-paste("KS统计量:",max(ks),sep="")
legend(0.3,0.2,c(kslable),2:8)






