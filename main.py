from tensorflow.keras.datasets import mnist
from one_way_neural_network import *
from convolution import *
test = Net(np.array([231,15*15,12*12,10]))
tru = 0
fals = 0
(trainX, trainy), (testX, testy) = mnist.load_data()
krenelI = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
krenel1 = [np.array([[-1,-1,-1],[1,1,1],[0,0,0]]),np.array([[-1,1,0],[-1,1,0],[-1,1,0]])]
krenel2 = [np.array([[0,0,0],[0,1,0],[0,0,0]]),np.array([[-1,-1,1],[-1,1,0],[1,0,0]]),np.array([[1,-1,-1],[0,1,-1],[0,0,1]])]
krenel3 = [np.array([[1,1,0,0,0],[1,1,0,0,0],[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0]]),np.array([[-1,-1,0,0,0],[-1,0,0,0,0],[0,0,0,1,1],[0,0,1,1,1],[0,0,1,1,1]]),np.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]])]
krenel3 = [np.array([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]),np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,1,1],[0,0,0,1,1]])]
for i in range(60000):
    print(i)
    x = trainX[i]#.reshape(28*28) #XD
    x = convolution5x5(krenelI,x)
    x = convolution5x5(krenelI,x)
    x1 = np.array([convolution3x3(i,x) for i in krenel1])
    x2 = np.array([[convolution3x3(i,j) for j in x1] for i in krenel1])
    x3 = np.array([[[convolution3x3(i,k) for k in j] for j in x2] for i in krenel1])
    x4 = np.array([[[[convolution5x5(i,t) for t in k] for k in j] for j in x3] for i in krenel3])
    x5 = np.array([[[[[convolution5x5(i,h) for h in t] for t in k] for k in j] for j in x4] for i in krenel3])
    x6 = x5.reshape(1152)[::5]#[0:9:-1]
    y = np.zeros(10)
    y[trainy[i]] = 1
    res = test.generate_output(x6)
    test.back_propagation(res,np.array(y,dtype=float))
    #print(np.power(res-y,2)/2)
    result = np.array([1 if res[i] > 0.5 else 0 for i in range(len(y))])
    print(res,"::",result,"::",y)
    #if result - y != 0:
    if i>50000:
        print(res,"::",result, " : ", y)
        if np.linalg.norm(result - y) == 0:
            tru +=1
        else:
            fals +=1
        #print(res , int(y))
print(float(tru)/float(tru+fals))
"""
n = 50
t = int(n/2)
l = n-t
x = np.array([float(i)/12 for i in range(-t,l)])
y = np.array([float(i)/12 for i in range(-t,l)])
z = np.array([[x[i%n],y[i//n],test.generate_output(np.array([x[i%n], y[i//n]]))[0]] for i in range(n**2)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(z[:,0], z[:,1], z[:,2], color='white', edgecolors='grey', alpha=0.5)
#ax.scatter(z[:,0], z[:,1], z[:,2], c='red')
plt.show()
"""