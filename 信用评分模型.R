setwd("/Users/ridono/Desktop/R")
#1、违约推断
develop<-read.csv("data.csv")
develop<-develop[,-1]
develop$default<-as.factor(develop$default)
#2 建立模型
#2.2 建立违约预测模型
#2.2.1 变量粗筛
names(develop);table(develop$default)
step2_1=develop
#根据业务理解生成更有意义的衍生变量

library(party)
set.seed(42)
crf<-cforest(default~.,control = cforest_unbiased(mtry = 2, ntree = 50), data=step2_1)
varimpt<-data.frame(varimp(crf))
c<-as.data.frame(varimpt)
#按照重要性进行排序，选择最重要的9个变量。这里看上去比较随意，如果要使用统计方法确定变量个数的话，需要使用IV这个统计量，此处不做演示
step2_2<-step2_1[,c("default","zm_score","risk_cnt","risk_score","org_cnt","search_cnt","call_times","hulu_score","online_time","age","nocall_rate")]
#2.2.2 变量细筛：
#也可以只做变量细筛，不过工作量会增大数倍
#library(devtools)
#install_github("riv","tomasgreif")
library(woe)
IV<-iv.mult(step2_2,"default",TRUE)
iv.plot.summary(IV)
step2_3=step2_2[,c("default","risk_cnt","zm_score","org_cnt","search_cnt","hulu_score","risk_score","call_times")]
summary(step2_3)

cor1<-cor(step2_3[,2:8])
library(corrplot)
corrplot(cor1,method = "number")

step2_3=step2_2[,c("default","risk_cnt","zm_score","org_cnt","hulu_score","risk_score","call_times")]
#不能只看统计量，还要仔细的察看每个变量的取值情况。一般建模数据是经过抽样的，大致在2000-5000，而且变量也经过了粗筛，因此用几个小时肯定可以把数据问题都找到。
head(step2_3,20)



#2.2.3 缺失值填补策略：
#注意，如果要评估模型的真实情况，需要将数据分为训练和验证集。验证集补缺只能使用训练集的统计量
d = sort(sample(nrow(step2_3), nrow(step2_3)*.6))
#select training sample
train<-step2_3[d,]
test<-step2_3[-d,]

#2.2.3_1 进行变量WOE转换
row.names(train) = seq(1,nrow(train))
iv.mult(train,"default",vars=c("risk_cnt"))
iv.plot.woe(iv.mult(train,"default",vars=c("risk_cnt"),summary=FALSE))

step2_2 <- iv.replace.woe(train,iv=iv.mult(train,"default"))

#2.2.3_1 构造逻辑回归模型
lg<-glm(default~.,family=binomial(link='logit'),data=train)
##
lg_both<-step(lg,direction = "both")
lg_forward<-step(lg,direction = "forward")
summary(lg)

#2.2.3 检验模型
#做出概率预测
logit<-predict(lg_both,test)
test$lg_both_p<-exp(logit)/(1+exp(logit))

logit<-predict(lg_forward,test)
test$lg_forward_p<-exp(logit)/(1+exp(logit))

test$out<-1
test[test$lg_both_p<0.5,]$out<-0
table(test$default,test$out)

#绘制ROC曲线
library(ROCR)
pred_both <- prediction(test$lg_both_p, test$default)
perf_both <- performance(pred_both,"tpr","fpr")
pred_forward <- prediction(test$lg_forward_p, test$default)
perf_forward <- performance(pred_forward,"tpr","fpr")

plot(perf_both,col='green',main="ROC of Models")
plot(perf_forward, col='blue',lty=2,add=TRUE);
abline(0,1,lty=2,col='red')

lr_m_auc<-round(as.numeric(performance(pred_both,'auc')@y.values),3)
lr_m_str<-paste("Mode_both-AUC:",lr_m_auc,sep="")
legend(0.5,0.55,c(lr_m_str),2:8)

lr_m_auc<-round(as.numeric(performance(pred_forward,'auc')@y.values),3)
lr_m_str<-paste("Mode_forward-AUC:",lr_m_auc,sep="")
legend(0.5,0.35,c(lr_m_str),2:8)

CrossTable(test$default, test$out, prop.chisq = FALSE, prop.c = F, 
           prop.r = F, dnn = c("actual", "predicted "))





