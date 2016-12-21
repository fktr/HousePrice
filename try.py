from math import sqrt
import math
import random

def diff(v1,v2):
    d_num=0
    d_str=0
    for i in range(len(v1)):
        if v1[i].isdigit() and v2[i].isdigit():
            d_num+=(int(v1[i])-int(v2[i]))**2
        elif v1[i]!=v2[i]:
            d_str+=1
    return sqrt(d_num),d_str

def get_similarity_list(data,v1):
    dlist_num=[]
    dlist_str=[]
    for i in range(len(data)):
        v2=data[i]['input']
        d_num,d_str=diff(v1,v2)
        dlist_num.append((d_num,i))
        dlist_str.append((d_str,i))
    dlist_num.sort()
    dlist_str.sort()
    return dlist_num,dlist_str

def gaussian(dist,sigma=10.0):
    return math.e**(-dist**2/(2*sigma**2))

def wknn(data,v1,k_num=10,k_str=10,wfun=gaussian,w_num=1,w_str=1):
    dlist_num,dlist_str=get_similarity_list(data,v1)
    weight_num=1
    price_num=0
    weight_str=1
    price_str=0
    for i in range(k_num):
        w=wfun(i)
        price_num+=data[dlist_num[i][1]]['result']*w
        weight_num+=w
    for i in range(k_str):
        w=wfun(i)
        price_str+=data[dlist_str[i][1]]['result']*w
        weight_str+=w
    price_num=price_num/weight_num
    price_str=price_str/weight_str
    return (price_str*w_str+price_num*w_num)/(w_str+w_num)

def divideset(data,rate=0.2):
    trainset=[]
    testset=[]
    for row in data:
        if random.random()<rate:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset,testset

def docost(data,v,max_times=100):
    if v is None:
        return 999999999
    error = 0
    for i in range(max_times):
        trainset, testset = divideset(data)
        for item in testset:
            guess = wknn(trainset, item['input'], k_num=v[0], k_str=v[1], w_num=v[2], w_str=v[3])
            error += (item['result'] - guess) ** 2
    return error / (max_times * len(testset))

def geneticoptmize(domain,costf,data,size=50,goodrate=0.2,mutprob=0.2,step=1,max_times=100):

    def mutate(v):
        i=random.randint(0,len(domain)-1)
        if random.random()<0.5 and v[i]>domain[i][0]:
            return v[0:i]+[v[i]-step]+v[i+1:]
        elif v[i]<domain[i][1]:
            return v[0:i]+[v[i]+step]+v[i+1:]

    def crossover(v1,v2):
        i=random.randint(1,len(domain)-1)
        return v1[0:i]+v2[i:]

    group=[]
    for i in range(size):
        v=[random.randint(domain[j][0],domain[j][1]) for j in range(len(domain))]
        group.append(v)
    goods=size*goodrate
    scores=None
    for i in range(max_times):
        scores=[(costf(data,v),v) for v in group]
        scores.sort()
        ranked=[v for (s,v) in scores]
        group=ranked[0:goods]

        while len(group)<size:
            if random.random()<mutprob:
                i=random.randint(0,goods-1)
                group.append(mutate(ranked[i]))
            else:
                i=random.randint(0,goods-1)
                j=random.randint(0,goods-1)
                group.append(crossover(ranked[i],ranked[j]))
    return scores[0][0],scores[0][1]


houses=[]
with open('train.csv') as f:
    f.readline()
    for line in f:
        row=line.strip('\n').split(',')
        houses.append({'input':[row[i] for i in range(1,len(row)-1)],'result':int(row[len(row)-1])})

domain=[(1,10)]*4
best,bestv=geneticoptmize(domain,docost,houses)
print(best,bestv)

test=[]
with open('test.csv') as f:
    f.readline()
    for line in f:
        row=line.strip('\n').split(',')
        test.append({'id':row[0],'input':[row[i] for i in range(1,len(row))]})

with open('result.csv','wt') as f:
    print("Id,SalePrice", file=f)
    for item in test:
        print('%s,%f' %(item['id'],wknn(houses,item['input'],k_num=bestv[0],k_str=bestv[1],w_num=bestv[2],w_str=bestv[3])), file=f)

'''
79个特征大约有35-36个为数字类型,43-44个为字符串类型
        a=0
        b=0
        for i in range(1,len(row)-1):
           if row[i].isdigit():
               a+=1
           else:
               b+=1
        print(a,b)
'''