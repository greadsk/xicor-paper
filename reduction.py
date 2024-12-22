# %%
import os 
os.getcwd()
print(os.getcwd())
## 获取当前工作目录

import pandas as pd
from pandas import DataFrame   #数据框的简写，本书采用的是该格式，直接使用DataFrame即可
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#pd.set_option('display.max_rows', None) #显示所有数据行 
pd.set_option('display.max_rows',10)     #共显示10行数据
pd.set_option('display.max_columns',None)# 设置查看列不省略
pd.set_option('display.precision',2)#设置后四位
pd.options.display.float_format='{:.2f}'.format # 全局设置二位小数

# %% [markdown]
# * demo1——标普500数据主成分降维后，主成分回归分析

# %% [markdown]
# 数据清洗，缺失值使用后一日的收盘价填充，整列缺失值或大量缺失值则删除。

# %%
#爬取标普500数据，12-2至12-10分钟级收盘价数据，创建demo数据库
import datetime as dt
import yfinance as yf

def get_data_tocsv(url=r'.\sp500tickers.xlsx'):
    msft = yf.Ticker('SPY')
    hist = msft.history(start="2024-12-2", end="2024-12-10", interval="1m")
    lie_all=hist['Close'].rename('SPY',inplace=True)
    df = pd.read_excel(url)
    for i in df.iloc[:,0]:
        msft=yf.Ticker(i)
        hist=msft.history(start="2024-12-2", end="2024-12-10", interval="1m")
        lie=hist['Close'].rename(i,inplace=True)
        lie_all=pd.concat([lie_all,lie],axis=1)
    lie_all.to_csv('sp500_data.csv')
    return lie_all
    
get_data_tocsv()

# %%
from sklearn.preprocessing import StandardScaler
data_origin=pd.read_csv(r'.\sp500_data.csv')
data_origin.rename(columns={data_origin.columns[0]: 'time'}, inplace=True)
data_origin.set_index(data_origin.columns[0], inplace=True)#设置时间为索引
data_origin.info()
#需要剔除的列 AZO,BKNG,NVR,POOL,BF.B,TPL,AIZ,BRK.B,CINF,ERIE,FDS,EG,MTD,BR,HUBB,JBHT,NDSN,SNA,GWW,FICO,TYL
need_to_drop=['AZO','BKNG','NVR','POOL','BF.B','TPL','AIZ','BRK.B','CINF','ERIE','FDS','EG','MTD','BR','HUBB','JBHT','NDSN','SNA','GWW','FICO','TYL']
data=data_origin.drop(need_to_drop,axis=1).bfill()#删除列和填充缺失值
missing_values = data.isnull().sum().sum()#查看是否还有缺失值
print(missing_values)
scaler = StandardScaler()#标准化
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data.head()#查看数据
y_data = data['SPY'] #分开自变量和因变量
x_data = data.drop('SPY', axis=1)
x_data.shape

# %%
from scipy.stats import pearsonr, spearmanr, kendalltau
from xicor.xicor import Xi
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import pearsonr, spearmanr, kendalltau
from xicor.xicor import Xi

# %% [markdown]
# * 模拟实验1 不同分布随机变量的度量情况

# %%
def comparison_xicor_difference_pdf(calculation ,title_name='各种分布假定下的xicor系数对sin函数的衡量',gap=5,n=50):
    beta=[];binomial=[];chisquare=[];exponential=[];f=[];gamma=[]
    geometric=[];gumbel=[];laplace=[];logistic =[];lognormal =[];
    negative_binomial=[];noncentral_chisquare=[];noncentral_f=[];
    norm=[];pareto=[];t=[];uniform=[];weibull=[];
    for i in [gap*a for a in range(n)]:
        x_beta=np.random.beta(a=2,b=2,size=i+2)
        x_binomial=np.random.binomial(n=10,p=0.5,size=i+2)
        x_chisquare=np.random.chisquare(df=5,size=i+2)
        x_exponential=np.random.exponential(scale=1,size=i+2)
        x_f=np.random.f(dfnum=2, dfden=2,size=i+2)
        x_gamma=np.random.gamma(shape=5,scale=1,size=i+2)
        x_geometric=np.random.geometric(p=0.5,size=i+2)
        x_gumbel=np.random.gumbel(loc=0,scale=1,size=i+2)
        x_laplace=np.random.laplace(loc=0,scale=1,size=i+2)
        x_logistic=np.random.logistic(loc=0,scale=1,size=i+2)
        x_lognormal=np.random.lognormal(mean=0,sigma=1,size=i+2)
        x_negative_binomial=np.random.negative_binomial(n=5,p=0.5,size=i+2)
        x_noncentral_chisquare=np.random.noncentral_chisquare(df=5,nonc=5,size=i+2)
        x_noncentral_f=np.random.noncentral_f(dfnum=5,dfden=5,nonc=5,size=i+2)
        x_norm=np.random.normal(loc=0,scale=1,size=i+2)
        x_pareto=np.random.pareto(a=5,size=i+2)
        x_t=np.random.standard_t(df=5,size=i+2)
        x_uniform=np.random.uniform(low=0,high=1,size=i+2)
        x_weibull=np.random.weibull(a=5,size=i+2)
        y_beta=calculation(x_beta)
        y_binomial=calculation(x_binomial)
        y_chisquare=calculation(x_chisquare)
        y_exponential=calculation(x_exponential)
        y_f=calculation(x_f)
        y_gamma=calculation(x_gamma)
        y_geometric=calculation(x_geometric)
        y_gumbel=calculation(x_gumbel)
        y_laplace=calculation(x_laplace)
        y_logistic=calculation(x_logistic)
        y_lognormal=calculation(x_lognormal)
        y_negative_binomial=calculation(x_negative_binomial)
        y_noncentral_chisquare=calculation(x_noncentral_chisquare)
        y_noncentral_f=calculation(x_noncentral_f)
        y_norm=calculation(x_norm)
        y_pareto=calculation(x_pareto)
        y_t=calculation(x_t)
        y_uniform=calculation(x_uniform)
        y_weibull=calculation(x_weibull)
        beta.append(max(Xi(list(x_beta),list(y_beta)).correlation,Xi(list(y_beta),list(x_beta)).correlation))
        binomial.append(max(Xi(list(x_binomial),list(y_binomial)).correlation,Xi(list(y_binomial),list(x_binomial)).correlation))
        chisquare.append(max(Xi(list(x_chisquare),list(y_chisquare)).correlation,Xi(list(y_chisquare),list(x_chisquare)).correlation))
        exponential.append(max(Xi(list(x_exponential),list(y_exponential)).correlation,Xi(list(y_exponential),list(x_exponential)).correlation))
        f.append(max(Xi(list(x_f),list(y_f)).correlation,Xi(list(y_f),list(x_f)).correlation))
        gamma.append(max(Xi(list(x_gamma),list(y_gamma)).correlation,Xi(list(y_gamma),list(x_gamma)).correlation))
        geometric.append(max(Xi(list(x_geometric),list(y_geometric)).correlation,Xi(list(y_geometric),list(x_geometric)).correlation))
        gumbel.append(max(Xi(list(x_gumbel),list(y_gumbel)).correlation,Xi(list(y_gumbel),list(x_gumbel)).correlation))
        laplace.append(max(Xi(list(x_laplace),list(y_laplace)).correlation,Xi(list(y_laplace),list(x_laplace)).correlation))
        logistic.append(max(Xi(list(x_logistic),list(y_logistic)).correlation,Xi(list(y_logistic),list(x_logistic)).correlation))
        lognormal.append(max(Xi(list(x_lognormal),list(y_lognormal)).correlation,Xi(list(y_lognormal),list(x_lognormal)).correlation))
        negative_binomial.append(max(Xi(list(x_negative_binomial),list(y_negative_binomial)).correlation,Xi(list(y_negative_binomial),list(x_negative_binomial)).correlation))
        noncentral_chisquare.append(max(Xi(list(x_noncentral_chisquare),list(y_noncentral_chisquare)).correlation,Xi(list(y_noncentral_chisquare),list(x_noncentral_chisquare)).correlation))
        noncentral_f.append(max(Xi(list(x_noncentral_f),list(y_noncentral_f)).correlation,Xi(list(y_noncentral_f),list(x_noncentral_f)).correlation))
        norm.append(max(Xi(list(x_norm),list(y_norm)).correlation,Xi(list(y_norm),list(x_norm)).correlation))
        pareto.append(max(Xi(list(x_pareto),list(y_pareto)).correlation,Xi(list(y_pareto),list(x_pareto)).correlation))
        t.append(max(Xi(list(x_t),list(y_t)).correlation,Xi(list(y_t),list(x_t)).correlation))
        uniform.append(max(Xi(list(x_uniform),list(y_uniform)).correlation,Xi(list(y_uniform),list(x_uniform)).correlation))
        weibull.append(max(Xi(list(x_weibull),list(y_weibull)).correlation,Xi(list(y_weibull),list(x_weibull)).correlation))

        
        
    
    plt.figure()
    plt.plot(range(n), beta, marker=None, label='beta')
    plt.plot(range(n), binomial, marker=None, label='binomial')
    plt.plot(range(n), chisquare, marker=None, label='chisquare')
    plt.plot(range(n), exponential, marker=None, label='exponential')
    plt.plot(range(n), f, marker=None, label='f')
    plt.plot(range(n), gamma, marker=None, label='gamma')
    plt.plot(range(n), geometric, marker=None, label='geometric')
    plt.plot(range(n), gumbel, marker=None, label='gumbel')
    plt.plot(range(n), laplace, marker=None, label='laplace')
    plt.plot(range(n), logistic, marker=None, label='logistic')
    plt.plot(range(n), lognormal, marker=None, label='lognormal')
    plt.plot(range(n), negative_binomial, marker=None, label='negative_binomial')
    plt.plot(range(n), noncentral_chisquare, marker=None, label='noncentral_chisquare')
    plt.plot(range(n), noncentral_f, marker=None, label='noncentral_f')
    plt.plot(range(n), norm, marker=None, label='norm')
    plt.plot(range(n), pareto, marker=None, label='pareto')
    plt.plot(range(n), t, marker=None, label='t')
    plt.plot(range(n), uniform, marker=None, label='uniform')
    plt.plot(range(n), weibull, marker=None, label='weibull')
    plt.title(title_name);plt.xlabel('样本数/{}'.format(gap));plt.ylabel('rho_Value');plt.grid(False)
    # 显示图例,网格（可选）
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()    
        
        
        
    
    

# %%
comparison_xicor_difference_pdf(lambda x: np.sin(x),n=50)

# %%
# #R语言的RDC 函数
# rdc <- function(x,y,k,s) {
#   x  <- cbind(apply(as.matrix(x),2,function(u) ecdf(u)(u)),1)
#   y  <- cbind(apply(as.matrix(y),2,function(u) ecdf(u)(u)),1)
#   wx <- matrix(rnorm(ncol(x)*k,0,s),ncol(x),k)
#   wy <- matrix(rnorm(ncol(y)*k,0,s),ncol(y),k)
#   cancor(cbind(cos(x%*%wx),sin(x%*%wx)), cbind(cos(y%*%wy),sin(y%*%wy)))$cor[1]
# }
# x <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
# y <- matrix(c(5, 6, 7, 8), nrow = 2, byrow = TRUE)
# result <- rdc(x, y, k = 5, s = 1)
# cat(sprintf("%.7f\n", result))
#随机相依系数函数实现
from scipy.stats import rankdata
from scipy.linalg import svd

def rdc(x, y, k, s):
    # Convert to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)
    
    # Apply ECDF to each column of x and y
    x_ecdf = np.apply_along_axis(lambda u: rankdata(u, method='max') / len(u), axis=0, arr=x)
    y_ecdf = np.apply_along_axis(lambda u: rankdata(u, method='max') / len(u), axis=0, arr=y)
    
    # Add a column of ones
    x = np.c_[x_ecdf, np.ones(x.shape[0])]
    y = np.c_[y_ecdf, np.ones(y.shape[0])]
    
    # Generate random matrices wx and wy
    wx = np.random.normal(0, s, (x.shape[1], k))
    wy = np.random.normal(0, s, (y.shape[1], k))
    
    # Compute the projections
    x_proj = np.cos(np.dot(x, wx)) + 1j * np.sin(np.dot(x, wx))
    y_proj = np.cos(np.dot(y, wy)) + 1j * np.sin(np.dot(y, wy))
    
    # Compute the canonical correlation
    Ux, Sx, Vx = svd(x_proj, full_matrices=False)
    Uy, Sy, Vy = svd(y_proj, full_matrices=False)
    
    # Compute the correlation matrix
    cor_matrix = np.dot(Ux.T, Uy)
    
    # Return the first canonical correlation coefficient
    return np.abs(cor_matrix).max()

# Example usage:
x = [[1, 2], [3, 4]]
y = [[5, 6], [7, 8]]
print(rdc(x, y, k=5, s=0))

# %% [markdown]
# * 模拟实验2

# %%
#衡量交互效应
def interaction_effect_three(pdf_type='difference',choose='x1',gap=5,min_=0,max_=100,n=50):
    x1_112=[];x1_113=[];x1_122=[];x1_133=[];x1_123=[];
    x2_112=[];x2_122=[];x2_123=[];x2_223=[];x2_233=[];
    x3_113=[];x3_123=[];x3_133=[];x3_223=[];x3_233=[];
    for i in [gap*a for a in range(n)]:
        if pdf_type=='difference':
            x1=np.linspace(min_,max_,i+2) #生成i+2个数据点
            x2=np.random.randint(min_,max_,i+2)
            x3=np.random.binomial(int(max_),0.5,i+2)
        elif pdf_type=='Bernoulli':
            x1=np.random.binomial(1,0.5,i+2) #生成i+2个数据点
            x2=np.random.binomial(1,0.5,i+2)
            x3=np.random.binomial(1,0.5,i+2)
        elif pdf_type=='binomial':
            x1=np.random.binomial(int(max_),0.5,i+2)
            x2=np.random.binomial(int(max_),0.5,i+2)
            x3=np.random.binomial(int(max_),0.5,i+2)
            
        x1x1x2=x1*x1*x2;x1x1x3=x1*x1*x3;x1x2x2=x1*x2*x2;x1x3x3=x1*x3*x3;
        x1x2x3=x1*x2*x3;x2x2x3=x2*x2*x3;x2x3x3=x2*x3*x3
        
        if choose=='x1':
            x1_112.append(max(Xi(list(x1),list(x1x1x2)).correlation,Xi(list(x1x1x2),list(x1)).correlation))
            x1_113.append(max(Xi(list(x1),list(x1x1x3)).correlation,Xi(list(x1x1x3),list(x1)).correlation))
            x1_122.append(max(Xi(list(x1),list(x1x2x2)).correlation,Xi(list(x1x2x2),list(x1)).correlation))
            x1_133.append(max(Xi(list(x1),list(x1x3x3)).correlation,Xi(list(x1x3x3),list(x1)).correlation))
            x1_123.append(max(Xi(list(x1),list(x1x2x3)).correlation,Xi(list(x1x2x3),list(x1)).correlation))
            
        if choose=='x2':
            x2_112.append(max(Xi(list(x2),list(x1x1x2)).correlation,Xi(list(x1x1x2),list(x2)).correlation))
            x2_122.append(max(Xi(list(x2),list(x1x2x2)).correlation,Xi(list(x1x2x2),list(x2)).correlation))
            x2_123.append(max(Xi(list(x2),list(x1x2x3)).correlation,Xi(list(x1x2x3),list(x2)).correlation))
            x2_223.append(max(Xi(list(x2),list(x2x2x3)).correlation,Xi(list(x2x2x3),list(x2)).correlation))
            x2_233.append(max(Xi(list(x2),list(x2x3x3)).correlation,Xi(list(x2x3x3),list(x2)).correlation))
         
        if choose=='x3':
            x3_113.append(max(Xi(list(x3),list(x1x1x3)).correlation,Xi(list(x1x1x3),list(x3)).correlation))
            x3_123.append(max(Xi(list(x3),list(x1x2x3)).correlation,Xi(list(x1x2x3),list(x3)).correlation))
            x3_133.append(max(Xi(list(x3),list(x1x3x3)).correlation,Xi(list(x1x3x3),list(x3)).correlation))
            x3_223.append(max(Xi(list(x3),list(x2x2x3)).correlation,Xi(list(x2x2x3),list(x3)).correlation))
            x3_233.append(max(Xi(list(x3),list(x2x3x3)).correlation,Xi(list(x2x3x3),list(x3)).correlation))
           
    if choose=='x1':
        y1=x1_112;y2=x1_113;y3=x1_122;y4=x1_133;y5=x1_123
        label1=r'$\xi \left( x_1,x_1x_1x_2 \right) $'
        label2=r'$\xi \left( x_1,x_1x_1x_3 \right) $'
        label3=r'$\xi \left( x_1,x_1x_2x_2 \right) $'
        label4=r'$\xi \left( x_1,x_1x_3x_3 \right) $'
        label5=r'$\xi \left( x_1,x_1x_2x_3 \right) $'
        
    if choose=='x2':
        y1=x2_112;y2=x2_122;y3=x2_123;y4=x2_223;y5=x2_233
        label1=r'$\xi \left( x_2,x_1x_1x_2 \right) $'
        label2=r'$\xi \left( x_2,x_1x_2x_2 \right) $'
        label3=r'$\xi \left( x_2,x_1x_2x_3 \right) $'
        label4=r'$\xi \left( x_2,x_2x_2x_3 \right) $'
        label5=r'$\xi \left( x_2,x_2x_3x_3 \right) $'
        
    if choose=='x3':
        y1=x3_113;y2=x3_123;y3=x3_133;y4=x3_223;y5=x3_233
        label1=r'$\xi \left( x_3,x_1x_1x_3 \right) $'
        label2=r'$\xi \left( x_3,x_1x_2x_3 \right) $'
        label3=r'$\xi \left( x_3,x_1x_3x_3 \right) $'
        label4=r'$\xi \left( x_3,x_2x_2x_3 \right) $'
        label5=r'$\xi \left( x_3,x_2x_3x_3 \right) $'
        
    plt.figure()
    plt.plot(range(n), y1, marker=None, label=label1)
    plt.plot(range(n), y2, marker=None, label=label2)
    plt.plot(range(n), y3, marker=None, label=label3)
    plt.plot(range(n), y4, marker=None, label=label4)
    plt.plot(range(n), y5, marker=None, label=label5)
    plt.title('{}的交互作用图'.format(choose));plt.xlabel('样本数/{}'.format(gap));plt.ylabel('rho_Value');plt.grid(False)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    

# %%
interaction_effect_three(pdf_type='difference',choose='x1',gap=5,min_=0,max_=100,n=500)#
interaction_effect_three(pdf_type='difference',choose='x2',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='difference',choose='x3',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='Bernoulli',choose='x1',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='Bernoulli',choose='x2',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='Bernoulli',choose='x3',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='binomial',choose='x1',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='binomial',choose='x2',gap=5,min_=0,max_=100,n=500)
interaction_effect_three(pdf_type='binomial',choose='x3',gap=5,min_=0,max_=100,n=500)

# %% [markdown]
# #　模拟实验３

# %%
#模拟实验3



def comparison_single_varible(calculation,title_name=None,gap=5,min_=0,max_=100,n=50):
    pearson1=[];spearman1=[];kendall1=[];Xi1=[];rdc1=[];
    for i in [gap*a for a in range(n)]:
        # x1=np.random.normal(0,1,i+2)
        x1=np.linspace(min_,max_,i+2) #生成i+2个数据点
        y1=calculation(x1)
        pearson,pearson_p=pearsonr(x1, y1)
        spearman,spearman_p=spearmanr(x1, y1)
        kendall,kendall_p=kendalltau(x1, y1)
        # rdcr=rdc(x1, y1,5,1)
        Xir=max(Xi(list(x1),list(y1)).correlation,Xi(list(y1),list(x1)).correlation)
        # Xir_p=Xi(list(x1),list(y1)).pval_asymptotic(ties=True)
        pearson1.append(pearson)
        spearman1.append(spearman)
        kendall1.append(kendall)
        # rdc1.append(rdcr)
        Xi1.append(Xir)
        
        
    plt.figure()
    plt.plot(range(n), pearson1, marker=None, label='pearson')# 绘制pearson的折线图,marker='o'or's'
    plt.plot(range(n), spearman1, marker=None, label='spearman')# 绘制spearman数据集的折线图
    plt.plot(range(n), kendall1, marker=None, label='kendall')# 绘制kendall数据集的折线图
    plt.plot(range(n), Xi1, marker=None, label='Xi') # 绘制xicor数据集的折线图
    # plt.plot(range(n), rdc1, marker=None, label='rdc') # 绘制rdc数据集的折线图
    plt.title(title_name);plt.xlabel('样本数/{}'.format(gap));plt.ylabel('rho_Value');plt.legend();plt.grid(False)
    # 显示图例,网格（可选）
    plt.show()
    
    
comparison_single_varible(lambda x:np.cumsum(x),title_name=r'$y\propto \sum{x}$')   #模拟一个自变量为x的函数，y=x1+x2+x3+...+xn

comparison_single_varible(lambda x:pow(x,2)+pow(x,3),title_name=r'$y\propto x^2+x^3$') # y=x^2+x^3
comparison_single_varible(lambda x:np.exp(x),title_name=r'$y\propto exp(x)$') #y=exp(x)
comparison_single_varible(lambda x:np.sin(x),title_name=r'$y\propto sinx$',n=50) #y=sinx 样本量=250
# comparison_single_varible(lambda x:np.sin(x),title_name=r'$y\propto sinx$',n=500) #y=sinx 样本量=2500
comparison_single_varible(lambda x:np.sin(x)/x,title_name=r'$y\propto \frac{\sin x}{x}$',min_=0.1,n=50) #y=sinx/x 样本量=250
# comparison_single_varible(lambda x:x/np.sin(x),title_name=r'$y\propto \frac{x}{\sin x}$',min_=0.1,max_=100,n=50) #y=x/sinx 样本量=250
comparison_single_varible(lambda x:x/np.sin(x),title_name=r'$y\propto \frac{x}{\sin x}$',min_=0.1,max_=100,n=500) #y=x/sinx 样本量=2500
comparison_single_varible(lambda x:x/np.sin(x),title_name=r'$y\propto \frac{x}{\sin x}$',min_=0.1,max_=10000,n=50) #y=x/sinx 样本量=2500
comparison_single_varible(lambda x:x/np.sin(x),title_name=r'$y\propto \frac{x}{\sin x}$',min_=0.1,max_=200,n=500) #y=x/sinx 样本量=2500
comparison_single_varible(lambda x:pow(x,x),title_name=r'$y\propto x^x$',min_=0.1,n=50) #y=x^x 样本量=250
comparison_single_varible(lambda x:(pow(x,2.5)+x*np.sin(x**2)+np.log(x+1)+np.exp(x)/x)/10,min_=0.0001,max_=7,title_name=r'$y\propto x^{2.5}+xsin(x^2)+\ln \left( x+1 \right) -\frac{exp(x)}{x}$',n=50) #y=复杂函数 样本量=250
comparison_single_varible(lambda x:np.random.randint(x),title_name='无关系时',min_=0.1,n=50) #无关样本，样本量=250
comparison_single_varible(lambda x:np.random.randint(x),title_name='无关系时',min_=0.1,n=500) #无关样本，样本量=2500

# %%
#离散随机变量的分段函数情况
def dicrete_cut(x):
    y=[]
    for i in range(len(x)):
        if x[i]<5: y.append(-1)
        elif x[i]<10: y.append(0)
        else: y.append(1)
    return y
#连续随机变量的分段函数情况
def continue_cut(x):
    y=[]
    for i in range(len(x)):
        if x[i]<5:y.append(-1)
        elif x[i]<10: y.append(np.exp(-x[i])+10)
        elif x[i]<15: y.append(np.sin(x[i]))
        else: y.append(3)
    return y
comparison_single_varible(lambda x : dicrete_cut(x),title_name=r'$y\propto -1\left( x<5 \right) \cup 0\left( 5\leqslant x<10 \right) \cup 1\left( 10\leqslant x \right) $',min_=0.1,max_=20,n=50) #离散分段函数，样本量=2500
comparison_single_varible(lambda x : continue_cut(x),title_name='连续的分段函数',min_=0.1,max_=20,n=50) #'连续的分段函数'，样本量=250
comparison_single_varible(lambda x :np.sin(pow(x,2)+np.sin(x)),title_name='',min_=0,max_=10,n=50) #'连续的分段函数'，样本量=250


# %% [markdown]
# # 降维分析 

# %% [markdown]
# ## <span style="font-size: 24px;"> 主成分分析 </span>

# %%

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 1.导入数据
X_scaled = x_data.to_numpy()

# 2. PCA降维
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# 主成分方向
components = pca.components_

# 3. 绘图
fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=120)


# 投影后数据散点图
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c='magenta', alpha=0.7, edgecolors='k')
ax[0].set_title("主成分后的数据")
ax[0].set_xlabel("Principal Component 1")
ax[0].set_ylabel("Principal Component 2")

# 主成分方向的可视化
for i, (comp, var) in enumerate(zip(components, pca.explained_variance_)):
    ax[1].arrow(0, 0, comp[0] * var, comp[1] * var, 
                color=f'C{i}', width=0.02, head_width=0.1)#, label=f'PC{i+1}'
ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c='lightgreen', alpha=0.7, edgecolors='k')
ax[1].legend()
ax[1].set_title("在原本空间上的主成分")
ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## <span style="font-size: 24px;"> 因子分析 </span>

# %%
from sklearn.decomposition import FactorAnalysis
import seaborn as sns

#1. 数据准备
data = x_data
columns = x_data.columns

# 2. 因子分析
fa = FactorAnalysis(n_components=6, random_state=42)
fa.fit(data)

# 提取因子载荷矩阵和因子得分
factor_loadings = fa.components_.T
factor_scores = fa.transform(data)

# 3. 可视化
plt.figure(figsize=(12, 6))

# 因子载荷热力图
plt.subplot(1, 2, 1)
sort_df = pd.DataFrame(factor_loadings.T,columns=columns ).T
sort_df=sort_df.sort_values(by=[0,1], ascending=[False,False]).T
sort_df_numpy=sort_df.to_numpy().T    
sns.heatmap(sort_df_numpy[:30,:], annot=True, cmap="YlGnBu", xticklabels=["Factor1", "Factor2"], yticklabels=sort_df.columns[:30])
plt.title("因子载荷热力图")
plt.xlabel("Factors")
plt.ylabel("Variables")

# 因子得分散点图
plt.subplot(1, 2, 2)
plt.scatter(factor_scores[:, 0], factor_scores[:, 1], c="tomato", alpha=0.7, edgecolor="k")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("因子得分散点图")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")

plt.tight_layout()
plt.show()

# %%
Communication_Services= ['GOOGL', 'GOOG', 'T', 'CHTR', 'CMCSA', 'EA', 'FOXA', 'FOX', 'IPG', 'LYV', 'MTCH', 'META', 'NFLX', 'NWSA', 'NWS', 'OMC', 'PARA',
                         'TMUS', 'TTWO', 'VZ', 'DIS', 'WBD']
Consumer_Discretionary= ['ABNB', 'AMZN', 'APTV', 'BBY', 'BWA', 'CZR', 'KMX', 'CCL', 'CMG', 'DHI', 'DRI', 'DECK', 'DPZ', 'EBAY', 'EXPE', 'F', 'GRMN',
                         'GM', 'GPC', 'HAS', 'HLT', 'HD', 'LVS', 'LEN', 'LKQ', 'LOW', 'LULU', 'MAR', 'MCD', 'MGM', 'MHK', 'NKE', 'NCLH', 'ORLY', 'PHM', 'RL', 
                         'ROST', 'RCL', 'SBUX', 'TPR', 'TSLA', 'TJX', 'TSCO', 'ULTA', 'WYNN', 'YUM']
Consumer_Staples=['MO', 'ADM', 'BG', 'CPB', 'CHD', 'CLX', 'KO', 'CL', 'CAG', 'STZ', 'COST', 'DG', 'DLTR', 'EL', 'GIS', 'HSY', 'HRL', 'SJM', 'K', 'KVUE', 'KDP', 
                  'KMB', 'KHC', 'KR', 'LW', 'MKC', 'TAP', 'MDLZ', 'MNST', 'PEP', 'PM', 'PG', 'SYY', 'TGT', 'TSN', 'WBA', 'WMT']
Energy=['APA', 'BKR', 'CVX', 'COP', 'CTRA', 'DVN', 'FANG', 'EOG', 'EQT', 'XOM', 'HAL', 'HES', 'KMI', 'MPC', 'OXY', 'OKE', 'PSX', 'SLB', 'TRGP', 'VLO', 'WMB']
Financials=['AFL', 'ALL', 'AXP', 'AIG', 'AMP', 'AON', 'ACGL', 'AJG', 'BAC', 'BLK', 'BX', 'BK', 'BRO', 'COF', 'CBOE', 'SCHW', 'CB', 'C', 'CFG', 'CME', 'CPAY', 
            'DFS', 'FIS', 'FITB', 'FI', 'BEN', 'GPN', 'GL', 'GS', 'HIG', 'HBAN', 'ICE', 'IVZ', 'JKHY', 'JPM', 'KEY', 'KKR', 'L', 'MTB', 'MKTX', 'MMC', 'MA', 'MET',
            'MCO', 'MS', 'MSCI', 'NDAQ', 'NTRS', 'PYPL', 'PNC', 'PFG', 'PGR', 'PRU', 'RJF', 'RF', 'SPGI', 'STT', 'SYF', 'TROW', 'TRV', 'TFC', 'USB', 'V', 'WRB', 'WFC', 'WTW']
Health_Care=['ABT', 'ABBV', 'A', 'ALGN', 'AMGN', 'BAX', 'BDX', 'TECH', 'BIIB', 'BSX', 'BMY', 'CAH', 'CTLT', 'COR', 'CNC', 'CRL', 'CI', 'COO', 'CVS', 'DHR', 'DVA', 
             'DXCM', 'EW', 'ELV', 'GEHC', 'GILD', 'HCA', 'HSIC', 'HOLX', 'HUM', 'IDXX', 'INCY', 'PODD', 'ISRG', 'IQV', 'JNJ', 'LH', 'LLY', 'MCK', 'MDT', 'MRK', 
             'MRNA', 'MOH', 'PFE', 'DGX', 'REGN', 'RMD', 'RVTY', 'SOLV', 'STE', 'SYK', 'TFX', 'TMO', 'UNH', 'UHS', 'VRTX', 'VTRS', 'WAT', 'WST', 'ZBH', 'ZTS']
Industrials=['MMM', 'AOS', 'ALLE', 'AMTM', 'AME', 'ADP', 'AXON', 'BA', 'BLDR', 'CHRW', 'CARR', 'CAT', 'CTAS', 'CPRT', 'CSX', 'CMI', 'DAY', 'DE', 'DAL', 'DOV', 
             'ETN', 'EMR', 'EFX', 'EXPD', 'FAST', 'FDX', 'FTV', 'GE', 'GEV', 'GNRC', 'GD', 'HON', 'HWM', 'HII', 'IEX', 'ITW', 'IR', 'J', 'JCI', 'LHX', 'LDOS', 
             'LMT', 'MAS', 'NSC', 'NOC', 'ODFL', 'OTIS', 'PCAR', 'PH', 'PAYX', 'PAYC', 'PNR', 'PWR', 'RSG', 'ROK', 'ROL', 'RTX', 'LUV', 'SWK', 'TXT', 'TT', 'TDG', 
             'UBER', 'UNP', 'UAL', 'UPS', 'URI', 'VLTO', 'VRSK', 'WAB', 'WM', 'XYL']
Information_Technology=[ 'ACN', 'ADBE', 'AMD', 'AKAM', 'APH', 'ADI', 'ANSS', 'AAPL', 'AMAT', 'ANET', 'ADSK', 'AVGO', 'CDNS', 'CDW', 'CSCO', 'CTSH', 'GLW', 'CRWD',
                        'DELL', 'ENPH', 'EPAM', 'FFIV', 'FSLR', 'FTNT', 'IT', 'GEN', 'GDDY', 'HPE', 'HPQ', 'IBM', 'INTC', 'INTU', 'JBL', 'JNPR', 'KEYS', 'KLAC', 
                        'LRCX', 'MCHP', 'MU', 'MSFT', 'MPWR', 'MSI', 'NTAP', 'NVDA', 'NXPI', 'ON', 'ORCL', 'PLTR', 'PANW', 'PTC', 'QRVO', 'QCOM', 'ROP', 'CRM', 
                        'STX', 'NOW', 'SWKS', 'SMCI', 'SNPS', 'TEL', 'TDY', 'TER', 'TXN', 'TRMB', 'VRSN', 'WDC', 'ZBRA']
Materials=['APD', 'ALB', 'AMCR', 'AVY', 'BALL', 'CE', 'CF', 'CTVA', 'DOW', 'DD', 'EMN', 'ECL', 'FMC', 'FCX', 'IFF', 'IP', 'LIN', 'LYB', 'MLM', 'MOS', 'NEM', 'NUE',
           'PKG', 'PPG', 'SHW', 'SW', 'STLD', 'VMC']
Real_Estate=['ARE', 'AMT', 'AVB', 'BXP', 'CPT', 'CBRE', 'CSGP', 'CCI', 'DLR', 'EQIX', 'EQR', 'ESS', 'EXR', 'FRT', 'DOC', 'HST', 'INVH', 'IRM', 'KIM', 'MAA', 'PLD', 
             'PSA', 'O', 'REG', 'SBAC', 'SPG', 'UDR', 'VTR', 'VICI', 'WELL', 'WY']
Utilities=['AES', 'LNT', 'AEE', 'AEP', 'AWK', 'ATO', 'CNP', 'CMS', 'ED', 'CEG', 'D', 'DTE', 'DUK', 'EIX', 'ETR', 'EVRG', 'ES', 'EXC', 'FE', 'NEE', 'NI', 'NRG',
           'PCG', 'PNW', 'PPL', 'PEG', 'SRE', 'SO', 'VST', 'WEC', 'XEL']
Industry={'Communication_Services':Communication_Services,'Consumer_Discretionary':Consumer_Discretionary,
          'Consumer_Staples':Consumer_Staples,'Energy':Energy, 
          'Financials':Financials,'Health_Care':Health_Care,'Industrials':Industrials,
          'Information_Technology':Information_Technology,'Materials':Materials,
          'Real_Estate':Real_Estate,'Utilities':Utilities}
industry_dict=Industry
# 计算最长的列表长度
max_length = max(len(lst) for lst in industry_dict.values())

# 填充 NaN 以使所有列表长度相同
for key in industry_dict:
    industry_dict[key] += [np.nan] * (max_length - len(industry_dict[key]))

# 转换为 DataFrame
df_industries = pd.DataFrame(industry_dict)
# 函数：根据公司名称查找行业
def find_company_industry(company_name, industry_dict):
    for industry, companies in industry_dict.items():
        if company_name in companies:
            return industry
    return None



#因子1的行业占比
sort_df = pd.DataFrame(factor_loadings.T,columns=columns ).T
sort_df=sort_df.sort_values(by=[0,1], ascending=[False,False]).T
sort_df_numpy=sort_df.to_numpy().T   
columns1=sort_df.columns[:30]

# 检索每个公司的行业
company_industry_map = {company: find_company_industry(company, industry_dict) for company in columns1}

# 输出结果
for company, industry in company_industry_map.items():
    print(f"公司 {company} 所在行业: {industry}")


# %% [markdown]
# ## <span style="font-size: 24px;"> 独立成分分析 </span>

# %%
from sklearn.decomposition import FastICA

# 数据准备
X = x_data.to_numpy()  # 观测信号
time=np.arange(len(X))

#  应用 ICA
ica = FastICA(n_components=2, random_state=42)
S_estimated = ica.fit_transform(X)  # 分离出的信号
A_estimated = ica.mixing_  # 估计的混合矩阵

# 绘图分析
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)


# 混合信号
axes[0, 0].plot(time, X[:, 0], color='green')
axes[0, 0].set_title('股票A')
axes[0, 1].plot(time, X[:, 99], color='purple')
axes[0, 1].set_title('股票B')

# 分离信号
axes[1, 0].plot(time, S_estimated[:, 0], color='orange')
axes[1, 0].set_title('提取特征A')
axes[1, 1].plot(time, S_estimated[:, 1], color='cyan')
axes[1, 1].set_title('提取特征B')

plt.suptitle('Independent Component Analysis (ICA) Results', fontsize=16)
plt.show()

# %% [markdown]
# ## <span style="font-size: 24px;"> 多维尺度分析 </span>

# %%
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
data=x_data

distance_matrix = pairwise_distances(data, metric='euclidean')
# MDS降维
mds = MDS(n_components=6, dissimilarity='precomputed', random_state=42)
low_dim_data = mds.fit_transform(distance_matrix)

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=120)

# 图1：MDS降维后的散点图
scatter = ax[0].scatter(low_dim_data[:, 0], low_dim_data[:, 1], cmap='viridis', s=50, edgecolor='k')
ax[0].set_title("MDS Projection (2D)", fontsize=14)
ax[0].set_xlabel("Component 1")
ax[0].set_ylabel("Component 2")
plt.colorbar(scatter, ax=ax[0], label="Cluster Labels")

# 图2：距离矩阵的热力图
im = ax[1].imshow(distance_matrix, cmap='hot', interpolation='nearest')
ax[1].set_title("Distance Matrix Heatmap", fontsize=14)
ax[1].set_xlabel("Sample Index")
ax[1].set_ylabel("Sample Index")
plt.colorbar(im, ax=ax[1], label="Distance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## <span style="font-size: 24px;"> t-SNE </span>

# %%
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

X=x_data

# 使用PCA降维到3D空间（便于对比）
pca = PCA(n_components=3)
X_pca1 = pca.fit_transform(X)

# 使用t-SNE降维到2D空间
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

fig = plt.figure(figsize=(16, 8))

# 原始PCA 3D降维
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter = ax1.scatter(X_pca1[:, 0], X_pca1[:, 1], X_pca1[:, 2], cmap='rainbow', s=10)
ax1.set_title('PCA 3D Visualization')
ax1.set_xlabel('PCA1')
ax1.set_ylabel('PCA2')
ax1.set_zlabel('PCA3')
legend1 = ax1.legend(*scatter.legend_elements(), title="Classes", loc="best")
ax1.add_artist(legend1)

# t-SNE 2D降维
ax2 = fig.add_subplot(1, 2, 2)
scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap='rainbow', s=10)
ax2.set_title('t-SNE 2D Visualization')
ax2.set_xlabel('t-SNE1')
ax2.set_ylabel('t-SNE2')
legend2 = ax2.legend(*scatter.legend_elements(), title="Classes", loc="best")
ax2.add_artist(legend2)

plt.tight_layout()
plt.show()

# %% [markdown]
# UMAP

# %%

import umap
# 第一步：生成数据集
X=x_data.to_numpy()

# 第二步：应用 UMAP 进行降维

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

# 第三步：可视化
fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=120)

#  UMAP 2D 嵌入
scatter = axs[0].scatter(embedding[:, 0], embedding[:, 1],  cmap='Spectral', s=5)
axs[0].set_title("UMAP 2D Embedding")
axs[0].set_xlabel("UMAP1")
axs[0].set_ylabel("UMAP2")

# UMAP 嵌入的密度图
sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], cmap="Reds", fill=True, ax=axs[1])
axs[1].set_title("Density of UMAP Embedding")
axs[1].set_xlabel("UMAP1")
axs[1].set_ylabel("UMAP2")

# 为 UMAP 图添加颜色条
cbar = fig.colorbar(scatter, ax=axs[1], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label("Color")

plt.tight_layout()
plt.show()

# %% [markdown]
# KPCA

# %%
from sklearn.decomposition import KernelPCA


X=x_data.to_numpy()

# 核PCA降维
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15,fit_inverse_transform=True)
X_kpca = kernel_pca.fit_transform(X)

#投影后数据可视化
plt.subplot(1, 2, 1)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], cmap='plasma', edgecolor='k')
plt.title('Kernel PCA Projection')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 核PCA没有显式的解释方差，因此直接基于特征值来估计
lambdas = kernel_pca.eigenvalues_
explained_variance_ratio = lambdas / np.sum(lambdas)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', color='red')
plt.title('Cumulative Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')

# 总体展示
plt.tight_layout()
plt.show()

# %% [markdown]
# 自编码器

# %%
import torch
import torch.nn as nn
import torch.optim as optim

data = torch.tensor(x_data.to_numpy(), dtype=torch.float32)

# 2. 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# 3. 初始化模型
input_dim = 482
latent_dim = 2
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
epochs = 100
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    x_hat, z = model(data)
    loss = criterion(x_hat, data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. 可视化分析
model.eval()
with torch.no_grad():
    x_hat, z = model(data)

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 图1：原始数据 vs 重建数据
axs[0].scatter(data[:, 0], data[:, 99], color='blue', alpha=0.5, label='Original Data')
axs[0].scatter(x_hat[:, 0], x_hat[:, 99], color='red', alpha=0.5, label='Reconstructed Data')
axs[0].set_title('原始数据和重构后的数据')
axs[0].legend()
axs[0].grid(False)

# 图2：潜在空间分布
axs[1].scatter(z[:, 0], z[:, 1], color='green', alpha=0.7)
axs[1].set_title('潜在空间分布')
axs[1].grid(False)

plt.tight_layout()
plt.show()

# %%
z.shape

# %% [markdown]
# # 比较各种方法下特征与y的xicor相关系数

# %%
def reduction_xicor(x,y):
    x=np.array(x);y=np.array(y)
    p=x.shape[1];rho=[]
    for i in range(p):
        rho.append(max(Xi(list(x[:,i]),list(y)).correlation,Xi(list(y),list(x[:,i])).correlation))
    return rho

# %%
def reduction_pearsonr(x,y):
    x=np.array(x);y=np.array(y)
    p=x.shape[1];rho=[]
    for i in range(p):
        rho.append(pearsonr(x[:,i],y)[0])
    return rho

# %%
reduction_xicor(X_pca,y_data)

# %%
reduction_pearsonr(X_pca,y_data)

# %%
reduction_xicor(factor_scores,y_data)

# %%
reduction_pearsonr(factor_scores,y_data)

# %%
ica = FastICA(n_components=6, random_state=42)
S_estimated1 = ica.fit_transform(X)  # 分离出的信号
reduction_xicor(S_estimated1,y_data)

# %%
reduction_pearsonr(S_estimated1,y_data)

# %%
reduction_xicor(low_dim_data,y_data)

# %%
reduction_pearsonr(low_dim_data,y_data)

# %%
tsne1 = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
X_tsne1 = tsne1.fit_transform(X)
reduction_xicor(X_tsne1,y_data)

# %%
reduction_pearsonr(X_tsne1,y_data)

# %%
reducer1 = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=6, random_state=42)
embedding1 = reducer1.fit_transform(X)
reduction_xicor(embedding1,y_data)

# %%
reduction_pearsonr(embedding1,y_data)

# %%
kernel_pca1 = KernelPCA(n_components=6, kernel='rbf', gamma=15,fit_inverse_transform=True)
X_kpca1 = kernel_pca1.fit_transform(X)
reduction_xicor(X_kpca1,y_data)

# %%
reduction_pearsonr(X_kpca1,y_data)

# %%
reduction_xicor(z,y_data)

# %%
reduction_pearsonr(z,y_data)


