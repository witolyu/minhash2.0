import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import itertools

from scipy.stats import binom

base2 = True # set true to represent delta in base 2, otherwise in base 10.
hockeyStickDivergence = False # set true to use HSD instead of CDF tail bound.
lite = False

def Sensitivity(s, k, nb):
    # return the probability of sensitivity s, with k iteration and size of honest party's input set being nb.
    return binom.pmf(s, k, 1.0 / (nb + 1))

def SensitivityList(k, nb, lam):
    # return a list of tuples of (sensitivity and probability), only keep those with greater than 2^-lam probability
    res = []
    for s in range(0, k + 1):
        res.append((s, Sensitivity(s, k, nb)))
        if 1 - binom.cdf(s, k, 1.0 / (nb + 1)) <2 ** (-(lam+1)):
            break
    return res

def BinomChernoffBound(k,n,p):
    # if k > n*p:
    #     return -1
    assert k <= n * p
    a = float(k)/n
    D_a_p = a * math.log(a/p, math.e) + (1-a) * math.log((1-a)/(1-p), math.e)
    return math.e**(-n*D_a_p)


# def success_p(min_value, num_of_trials):
#     # Return the success probability of a Bernoulli trial capturing the probability that all non intersection items are
#     # hashed to a greater value than minimum across intersected item.
#     return (1-min_value) ** num_of_trials

def NumGoodHashes(na, nb, ni, k,lam):
    # this return the number of good hashes given a list of theta:
    # (1) it hashed intersected input to minimum.
    # (2) it hashes to within range of [1/2-theta, 1/2+theta].

    res = []

    theta_list = [0.02 * i for i in range(24, 0,-1)]

    k_good = k

    for theta in theta_list:
        # this is the success probability of the good hash event for each iteration.
        p = ((1/2+theta)**(na/(nb-ni)) - (1/2-theta)**(na/(nb-ni))) * ni/na

        while True:
            if k_good == 0:
                break
            if k_good >= p * k:
                # As we want to lower bound (with negligible probability), k_good should be less than expected.
                k_good -= 1
                continue

            # print(theta,k_good,k,p)
            # tail = BinomChernoffBound(k_good,k,p)
            tail = binom.cdf(k_good, k, p)

            if tail <= 2**(-lam):
                res.append((theta, k_good))
                break
            else:
                k_good -= 1

        # print(theta, k_good,p)

    return res

def NumGoodHashesGraph(na, nb, ni, k, theta_list):
    # this return the number of good hashes given a list of theta:
    # (1) it hashed intersected input to minimum.
    # (2) it hashes to within range of [1/2-theta, 1/2+theta].

    theta_list.sort(reverse=True)

    k_good = k

    x_list = []
    y_list = []

    # Iterate through different thetas...

    for theta in theta_list:
        p = ((1/2+theta)**(na/(nb-ni)) - (1/2-theta)**(na/(nb-ni))) * ni/na

        while True:
            if k_good == 0:
                break
            if k_good >= p * k:
                k_good -= 1
                continue

            # print(theta,k_good,k,p)
            tail = BinomChernoffBound(k_good,k,p)

            if tail <= 2**(-40):
                break
            else:
                k_good -= 1

        # print(theta, k_good,p)
        x_list.append(theta)
        y_list.append(k_good)
    print(x_list, y_list)
    plt.plot(x_list,y_list, 'ro')
    plt.axis([0, 0.5, 0, 250])
    plt.xlabel('Theta')
    plt.ylabel('# of "good" hash functions')
    plt.suptitle('n_A = {}, n_B = {}, n_I = {}, k = {}'.format(na,nb,ni,k))
    plt.show()

# Poisson binomial mechanism, return delta for given epsilon and k.

def BinomMechanism(p, k ,eps,s):
    e_eps = math.e ** eps
    e_eps_s = math.e**(eps/s)
    q = 1-p

    if hockeyStickDivergence:
        # FIXME: Implement Binary search and Hockey Stick Divergence
        l = math.ceil((k * p + s * p + s * q * e_eps_s) / (e_eps_s * q + p))
        l_delta = binom.cdf(l - 1, k, p)

        # right tail
        r = math.floor((k * p * e_eps_s) / (q + e_eps_s * p))
        r_delta = 1 - binom.cdf(r, k, p)
        # binoma
        # left = s
        # right = math.floor(p*k)-1
        # assert left<right
        # l = -1
        #
        # while left <= right:
        #     mid = (left + right) // 2
        #
        #     if binom.pdf(mid,k,p)/binom.pdf(mid-s,k,p) > e_eps:
        #         left = mid + 1
        #     else:
        #         l = mid
        #         right = mid - 1
        #
        #
        # l_delta = binom.cdf(l - 1, k, p) - e_eps_s *




    else:
    # left tail, use a looser bound compared to the paper, without assume p < 1/2.
        l = math.ceil((k*p+ s*p + s*q*e_eps_s)/(e_eps_s * q + p))
        l_delta = binom.cdf(l-1, k, p)

        # right tail
        r = math.floor((k*p*e_eps_s)/(q + e_eps_s* p))
        r_delta = 1-binom.cdf(r,k,p)

    # print(l,l_delta,r,r_delta)
    #
    # if l_delta >= r_delta:
    #     print("left tail")
    # else:
    #     print("right tail")
    return max(l_delta,r_delta)


def PBinomMechansim(na,nb,ni, k, eps,s, lam, mode):
    # mode = 0: throw away half; mode = 1: brute force; mode = 2: ratio between upper and lower bound.
    theta_k_list = NumGoodHashes(na, nb, ni, k, lam)
    min_delta = 1
    for theta, k_good in theta_k_list:
        k_good_half = math.floor(k_good/2)
        p = 0.5 - theta
        delta = BinomMechanism(p, k_good_half,eps,s)
        min_delta = min(min_delta, delta)

    return min_delta

# This compute the "real delta", i.e., the weighted sum of delta for different sensitivity, as well as a failure rate.
def MinhashPM(na, nb, ni, k, eps,lam, mode):
    # calculate sensitivity

    delta = 0

    total_w = 0

    s_list = SensitivityList(k, nb, lam)
    #
    # for s,w in s_list:
    #     delta += w * (PBinomMechansim(na,nb,ni, k, eps,s, lam, mode) + 1/(2**lam))

    # optimize the failure probability to claim good hashes, no need to make this extremely small, when delta term is large
    for s,w in s_list:
        total_w += w

        if s == 0:
            continue

        best_delta = 1
        if lite:
            best_delta = min(best_delta, PBinomMechansim(na, nb, ni, k - s, eps, s, lam, mode) + 1 / (2 ** lam))
        else:
            for l in range(20, lam+1,2):
                best_delta = min(best_delta,PBinomMechansim(na,nb,ni, k-s, eps,s, l, mode)+ 1/(2**l) )
        delta += w * best_delta

    delta += (1-total_w)

    return delta

# Compute dealta for Binomial Mechanism.
# This compute the "real delta", i.e., the weighted sum of delta for different sensitivity, as well as a failure rate.
def MinhashBM(na, nb, ni, k, eps,lam, mode):
    # Jaccard Similarity

    J = ni/ (na + nb - ni)

    # calculate sensitivity

    delta = 0

    total_w = 0

    s_list = SensitivityList(k, nb, lam)
    print(s_list)
    #
    # for s,w in s_list:
    #     delta += w * (PBinomMechansim(na,nb,ni, k, eps,s, lam, mode) + 1/(2**lam))

    # optimize the failure probability to claim good hashes, no need to make this extremely small, when delta term is large
    for s,w in s_list:
        total_w += w

        if s == 0:
            continue

        # best_delta = 1
        #
        # for l in range(20, lam+2):
        #     best_delta = min(best_delta,PBinomMechansim(na,nb,ni, k-s, eps,s, l, mode)+ 1/(2**l) )
        # delta += w * best_delta

        delta += w * BinomMechanism(J,k,eps,s)
        print(delta)
    delta += 2 ** (-(lam+1))
    return delta

def MinhashGraphBinom(na, nb, ni_list, k_list, eps_list,lam):

    for ni,k in itertools.product(ni_list,k_list):
        row = []
        for eps in eps_list:
            row.append(MinhashBM(na,nb,ni,k,eps,lam, 0))
        plt.plot(eps_list, row, label = "$n_I$ = {}, k = {}".format(ni, k))

        print('n_I = {}, k = {} completed'.format(ni, k))

    # plt.axis([0, max(eps_list)+1,])
    plt.xticks(np.arange(min(eps_list), max(eps_list) + 0.25, 0.25))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.ylabel(r'$\delta$',rotation=0, fontsize=18)
    if base2:
        plt.yscale('log', base=2)
        plt.yticks(np.array([2 ** (-1 * i) for i in range(16, 64,8)]))
    else:
        plt.yscale('log')
        plt.yticks(np.array([10**(-1 * i) for i in range(6,13)]))
    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    plt.show()

def MinhashGraphPBinom(na, nb, ni_list, k_list, eps_list,lam):

    for ni,k in itertools.product(ni_list,k_list):
        row = []
        for eps in eps_list:
            row.append(MinhashPM(na,nb,ni,k,eps,lam, 0))
        plt.plot(eps_list, row, label = "$n_I$ = {}, k = {}".format(ni, k))

        print('n_I = {}, k = {} completed'.format(ni, k))

    # plt.axis([0, max(eps_list)+1,])
    plt.xticks(np.arange(min(eps_list), max(eps_list) + 0.25, 0.25))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.ylabel(r'$\delta$',rotation=0, fontsize=18)
    if base2:
        plt.yscale('log', base=2)
        plt.yticks(np.array([2 ** (-1 * i) for i in range(16, 45,4)]))
    else:
        plt.yscale('log')
        plt.yticks(np.array([10**(-1 * i) for i in range(6,13)]))
    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    plt.show()

def MinhashPMK(na, nb, J, eps,delta, lam, mode):
    # Given the jaccard index, compute the number of hashes needed to achieve set eps delta.

    # find the smallest K that satisfies eps, delta.
    ni = math.floor((na+nb)*J/(1+J))

    precision = 100

    low = max(int(100/ precision),1)
    high = int(10000/ precision)

    print("Testing K = {}".format(high * precision))
    if MinhashPM(na, nb, ni, high* precision, eps, lam, mode) > delta:
        return np.nan
    # The binary search make contains small errors, as we may not be dealing with strictly decreasing function.
    # But it should be fine.
    while low < high:
        mid = (low + high) //2
        print("Testing K = {}".format(mid * precision))
        if MinhashPM(na, nb, ni, mid * precision, eps, lam, mode) < delta:
            high = mid
        else:
            low = mid + 1

    assert low == high
    return high* precision
def MinhashGraphPBinomJaccardVsK(na, nb, eps_list, delta_list, lam):

    J_list = [0.05 * i for i in range(1,20)]
    # J_list = [0.25 * i for i in range(1, 4)]
    # J_list = [0.1 * i for i in range(1, 10)]
    res = []

    for eps, delta in itertools.product(eps_list,delta_list):
        print("Computing for eps = {} and delta = {}".format(eps,delta))
        row = []
        for J in J_list:
            row.append(MinhashPMK(na,nb,J,eps,delta,lam, 0))
            print("J = {} completed.".format(J))
        plt.plot(J_list, row, 'o-', label = r"$\epsilon$ = %d $\delta$ = $2^{%d}$" % (eps, int(math.log(delta,2))))
        res.append(row)
        print('eps = {}, delta = {} completed'.format(eps, delta))

    # Since it is time-consuming, therefore print the output.
    print(res)

    plt.xticks(np.arange(0.05, 1, 0.05))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'Jaccard Index', fontsize=18)
    plt.ylabel(r'Number of Iterations', fontsize=18)

    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    plt.show()


def MinhashBMK(na, nb, J, eps,delta, lam, mode):
    # Given the jaccard index, compute the number of hashes needed to achieve set eps delta.

    # find the smallest K that satisfies eps, delta.
    ni = math.floor((na+nb)*J/(1+J))

    precision = 10

    low = max(int(10/ precision),1)
    high = int(500/ precision)

    print("Testing K = {}".format(high * precision))
    if MinhashBM(na, nb, ni, high* precision, eps, lam, mode) > delta:
        return np.nan
    # The binary search make contains small errors, as we may not be dealing with strictly decreasing function.
    # But it should be fine.
    while low < high:
        mid = (low + high) //2
        print("Testing K = {}".format(mid * precision))
        if MinhashBM(na, nb, ni, mid * precision, eps, lam, mode) < delta:
            high = mid
        else:
            low = mid + 1

    assert low == high
    return high* precision

def MinhashGraphBinomJaccardVsK(na, nb, eps_list, delta_list, lam):

    J_list = [0.05 * i for i in range(1,20)]
    # J_list = [0.25 * i for i in range(1, 4)]
    # J_list = [0.1 * i for i in range(1, 10)]
    res = []

    for eps, delta in itertools.product(eps_list,delta_list):
        print("Computing for eps = {} and delta = {}".format(eps,delta))
        row = []
        for J in J_list:
            row.append(MinhashBMK(na,nb,J,eps,delta,lam, 0))
            print("J = {} completed.".format(J))
        plt.plot(J_list, row, 'o-', label = r"$\epsilon$ = %f $\delta$ = $2^{%d}$" % (eps, int(math.log(delta,2))))
        res.append(row)
        print('eps = {}, delta = {} completed'.format(eps, delta))

    # Since it is time-consuming, therefore print the output.
    print(res)

    plt.xticks(np.arange(0.05, 1, 0.05))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'Jaccard Index', fontsize=18)
    plt.ylabel(r'Number of Iterations', fontsize=18)

    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # MinhashGraphPBinom(1000000, 1000000, [250000,500000,750000], [1000,2000],[1+i*0.25 for i in range(0,9)],50)


    # MinhashGraphPBinomJaccardVsK(1000000, 1000000,  [1,2], [2**(-20),2**(-30),2**(-40)], 40)

    # lambda should be set to be smaller than the lg2 of the last of the delta list.
    MinhashGraphPBinomJaccardVsK(100000, 100000, [1, 2], [2 ** (-20), 2 ** (-30), 2 ** (-40)], 44)

    # MinhashGraphBinom(1000000, 1000000,[250000,500000,750000], [100,200],[0.5+i*0.25 for i in range(0,8)],60)

    # MinhashGraphBinomJaccardVsK(1000000, 1000000,  [1,2], [2**(-30),2**(-40),2**(-50)], 60)

    # MinhashGraphBinomJaccardVsK(100000, 100000,  [1,2], [2**(-30),2**(-40),2**(-50)], 60)




   # print(BinomMechanism(0.6, 200 ,2,1),BinomMechanism(0.6, 200 ,2.5,1))

    # print(MinhashPMK(100000, 100000, 0.1, 1, 2**(-20), 40, 0))



    # for k in range(100,10000,200):
    #     print(k,MinhashPM(100000, 100000,18181, k, 1, 40, 0))
    #
    #     na, nb, ni, k, eps, lam, mode

