import numpy as np
#from vendor.perlin import perlin
from perlin_noise import PerlinNoise


def main():
    import matplotlib.pyplot as plt

    def pits(x):
        return np.power(np.tanh(90*(x-0.25)),2)*np.power(np.tanh(50*(x-0.75)),2)*np.power(np.tanh(60*(x-0.93)),2)*np.power(np.tanh(55*(x-0.09)),2)#*np.power(np.tanh(120*(x-0.85)),2)
    
    def pits_edge(x):
        return np.power(np.tanh(50*(x)),2)*np.power(np.tanh(50*(x-1)),2)*np.power(np.tanh(30*(x-0.5)),2)
    
    def baseLoss(x):
        return np.power(10*(x-0.5),2)*pits(x)
    

    shape = (2048)  # Specify the size of the noise grid
    frequency=1

    

    noise = PerlinNoise(octaves=1, seed=1)

    lossLandscape = baseLoss(np.linspace(0,1,shape))
    
    layers=10

    for k in range(layers):
        for i in range(shape):
            lossLandscape[i]+=4*noise((4*k+3)*i/shape*frequency)/(k+1)*pits(i/shape)*pits_edge(i/shape)


    time = np.linspace(0,1,len(lossLandscape))

    def cn(arr,n):
        c = arr*np.exp(-1j*2*n*np.pi*time)
        return c.sum()/c.size

    def f(arr_cn, t):
        f = np.array([2*arr_cn[i+1]*np.exp(1j*2*(i+1)*np.pi*t) for i in range(len(arr_cn)-1)])
        return np.sum(f, axis=0)
    
    def fprime(arr_cn, t):
        f = np.array([2*arr_cn[i+1]*np.exp(1j*2*(i+1)*np.pi*t)*1j*2*(i+1)*np.pi for i in range(len(arr_cn)-1)])
        return np.sum(f, axis=0)
    
    def fintegral(arr_cn, t):
        f = np.array([2*arr_cn[i+1]*np.exp(1j*2*(i+1)*np.pi*t)/(1j*2*(i+1)*np.pi) for i in range(len(arr_cn)-1)])
        return np.sum(f, axis=0)

    common_x = np.linspace(0, 1, len(lossLandscape))

    lossLandscape_cn = np.array([cn(lossLandscape,i) for i in range(0,100)])
    FLossLandscape = f(lossLandscape_cn,time).real

    # Perfect epsilon ball
    epsilon = 0.03
    print("calculating epsilon ball gradient")
    epsilonBallLandscape = np.max([f(lossLandscape_cn,np.linspace(time-epsilon,time+epsilon, 100)).real], axis=1).reshape(-1)

    #SAM/Adversarial/PGD
    
    print("calculating SAM effective gradient")
    EffTime = time+epsilon*np.sign(fprime(lossLandscape_cn, time).real)
        
    EffGradSAM = fprime(lossLandscape_cn, EffTime).real
    # EffGradSAM = np.clip(EffGradSAM, -100, 100)
    EffGradSAM_cn = np.array([cn(EffGradSAM,i) for i in range(0,100)])
    FLossLandscapeSAM = np.array([fintegral(EffGradSAM_cn,t).real for t in time])   
    FLossLandscapeSAM = f(lossLandscape_cn, EffTime).real

    #SAM PGD
    print("calculating SAM PGD effective gradient")
    n = 20
    alpha = epsilon/20
    EffPGDTime = time
    for i in range(n):
        EffPGDTime = EffPGDTime+alpha*np.sign(fprime(lossLandscape_cn, EffPGDTime).real)

    EffGradPGD = fprime(lossLandscape_cn, EffPGDTime).real
    PGD_lossLandscape_cn = np.array([cn(EffGradPGD,i) for i in range(0,100)])
    FLossLandscapePGD = np.array([fintegral(PGD_lossLandscape_cn,t).real for t in time]) 
    FLossLandscapePGD = f(lossLandscape_cn, EffPGDTime).real
     

    epsilonBallLandscape -= min(FLossLandscape)
    FLossLandscapeSAM -= min(FLossLandscape)
    FLossLandscapePGD -= min(FLossLandscape)
    FLossLandscape -= min(FLossLandscape)

    ReconstructedTrue = fprime(lossLandscape_cn, time).real
    RecCN = np.array([cn(ReconstructedTrue,i) for i in range(0,100)])
    ReconstructedTrue = fintegral(RecCN, time).real
    ReconstructedTrue -= min(ReconstructedTrue)
    

    plt.figure(figsize=(10, 6))
    # plt.plot(common_x, lossLandscape, label='True Loss Landscape', linewidth=2)
    plt.plot(common_x, FLossLandscape, label='True Loss Landscape', linewidth=2)
    # plt.plot(common_x, ReconstructedTrue, label='x', linewidth=2)
    plt.plot(common_x, epsilonBallLandscape, label=r'Perfect ρ-Ball Landscape', linewidth=2)
    plt.plot(common_x, FLossLandscapeSAM, label='Effective Loss Landscape (SAM)', linewidth=2)
    plt.plot(common_x, FLossLandscapePGD, label='Effective Loss Landscape (SAM PGD) (n=20)', linewidth=2)
    plt.xlabel('Weight θ', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Comparison of Loss Landscapes ρ=0.03', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('C:/Users/franc/Desktop/temp/PerlinTest.png', dpi=300)
    plt.close()

    

if __name__ == "__main__":
    main()


