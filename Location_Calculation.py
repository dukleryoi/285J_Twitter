import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

(W, H) = pickle.load(open('NMF_500_topics_WH.pkl','rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))
print(W.shape)
print(H.shape)
Spatial = pickle.load(open('pandas_data.pkl','rb'))
Topics = W.argmax(axis=1)#Assigns a topic to each tweet
Spatial["topics"] = Topics#data frame containing the valuable columns from raw data
Spatial = Spatial[Spatial["latitude"]>41]#removed one outlier with very low lattitude
maxlat = Spatial["latitude"].max()
minlat = Spatial["latitude"].min()
maxlong = Spatial["longitude"].max()
minlong = Spatial["longitude"].min()
print(maxlat, minlat)
print(maxlong, minlong)
Spatial["latitude"] = (Spatial["latitude"]-minlat)/(maxlat-minlat)#we normalize the latatitudal and longitudal data of the tweets to be between 0 and 1
Spatial["longitude"] = (Spatial["longitude"]-minlong)/(maxlong-minlong)
MSD_List = []#stores the Mean Square Distance between all the tweets in  agiven topic
Topics_Size = []
Spatial = Spatial[Spatial["gps_precision"] == 10.0]#taking gonly location accurete tweets
for T in range(0,500):#Mean Square Distance Calculation: we want to find the sum of the square of the
    x = (Spatial[Spatial["topics"] == T])# euclidean pairwise distances between all the tweets in each topic divided by the size of the topic (K)
    a = x["latitude"]
    b = x["longitude"]
    K =len(a)
    Topics_Size.append(K)
    X = np.array(a)#we calculate x axis pairwise square distances
    Y = np.array(b)#and then seperatley y axis distances
    MSD = (2/((K+1)**2))*((K*np.dot(X.T, X) - (X.sum())**2)+(K*np.dot(Y.T, Y) - (Y.sum())**2))#I am almost 100% this is correct but tell me if you guys see anything alarming
    MSD_List.append(MSD)#in the above, we used K+1 instead of K (Where K is the size of the topic) in the denomenator to make  sure we ar enot dividing by 0 for the empty topics
df = pd.DataFrame(columns = ( "Length", "MSD"))#save the topics size and MSD into a data frame
df["MSD"] = MSD_List
df["Length"] = Topics_Size

#next we want to create a "density function" of the tweets for each topic over the city of barcelona in order to calculate other location statistics
#to do this we partition our location grid of [0,1]x[0,1] into LxL squares (here L = 100) we create matrices for each of the 500 topics to store the
#the number of tweets in each of the squares, we later divide the matrix by the total number of tweets in the topic to get a probibistic matrix of
# where are the tweets distrbuted for each topic


L =100#L can be refined to 500 or even 1000
ArrayList = []
for T in range(0,500):#F density calculation
    A = np.zeros((L,L))
    G = Spatial[Spatial["topics"] == T]
    Glong = G["longitude"].tolist()
    Glat = G["latitude"].tolist()
    N = len(Glat)
    for i in range(0,N):
        x = int((Glong[i]*(L-1*10**(-12))))#subtracted 10^-12 from L which is neglible but needed to make sure that the one tweet with the maximal longitude
        y = int((Glat[i]*(L-1*10**(-12))))#(which is equal to 1 once normalized) was not causing an out of bounds error when used as an index for the array
        A[x,y] = A[x,y]+ 1
    ArrayList.append(A/N)

#once we created the denisty function array we calculate the information theoretic entropy associated with the probablity distribution
#as well as a fractional L^0.5 norm, normalized by the usual L^1 norm of the distribution
MetricList= []
EntropyList = []
for X in ArrayList:
    L_P = ((np.sqrt(X)).sum()*(1/(L**2)))**2
    L_1 = (X.sum()+1)*(1/(L**2))
    Final_L = L_P/L_1# L^0.5 norm divided by L^1 norm
    MetricList.append(Final_L)
    Log = np.log(X)#Log(P) matrix
    Log = np.nan_to_num(Log)# removes enteries where P was 0 hence Log(P)  was undefined
    EntropyMatrix = np.multiply(X,Log)
    Ent = -EntropyMatrix.sum()#sum of P*log(P) where P is the probablity that for a certain topic the tweets are in the specific square
    EntropyList.append(Ent)
df["L0.5"]= MetricList
df["Entropy"] = EntropyList
df = df.sort_values(by = "MSD")
df = df[df["Length"]>0]#removes empty topics
print(df.head(500))
#plt.plot(df["MSD"], df["L0.5"])
plt.scatter(df["L0.5"], df["Entropy"])
plt.show()