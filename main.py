import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import itertools

from scipy.stats import binom

def sensitivity(s, k, nb):
    # return the probability of sensitivity s, with k iteration and size of honest party's input set being nb.
    return binom.pmf(s, k, 1.0 / (nb + 1))

def sensitivity_list(k, nb, lam):
    # return a list of tuples of (sensitivity and probability), only keep those with greater than 2^-lam probability
    res = []
    for s in range(0, k + 1):
        res.append((s, sensitivity(s, k, nb)))
        if binom.cdf(s, k, 1.0 / (nb + 1)) > 1 - 2 ** (-lam):
            break
    return res

def chernoff_bound(k,n,p):
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

def num_good_hashes(na, nb, ni, k,lam):
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
            # tail = chernoff_bound(k_good,k,p)
            tail = binom.cdf(k_good, k, p)

            if tail <= 2**(-lam):
                res.append((theta, k_good))
                break
            else:
                k_good -= 1

        # print(theta, k_good,p)

    return res

def num_good_hashes_graph(na, nb, ni, k, theta_list):
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
            tail = chernoff_bound(k_good,k,p)

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


def Binom_Mechanism(p, k ,eps,s):
    e_eps_s = math.e**(eps/s)
    q = 1-p

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


def PBinom_Mechansim(na,nb,ni, k, eps,s, lam, mode):
    # mode = 0: throw away half; mode = 1: brute force; mode = 2: ratio between upper and lower bound.
    theta_k_list = num_good_hashes(na, nb, ni, k, lam)
    min_delta = 1
    for theta, k_good in theta_k_list:
        k_good_half = math.floor(k_good/2)
        p = 0.5 - theta
        delta = Binom_Mechanism(p, k_good_half,eps,s)
        min_delta = min(min_delta, delta)

    return min_delta

# This compute the "real delta", i.e., the weighted sum of delta for different sensitivity, as well as a failure rate.
def minhash_PM(na, nb, ni, k, eps,lam, mode):
    # calculate sensitivity

    delta = 0

    total_w = 0

    s_list = sensitivity_list(k, nb, lam)
    #
    # for s,w in s_list:
    #     delta += w * (PBinom_Mechansim(na,nb,ni, k, eps,s, lam, mode) + 1/(2**lam))

    # optimize the failure probability to claim good hashes, no need to make this extremely small, when delta term is large
    for s,w in s_list:
        total_w += w

        if s == 0:
            continue

        best_delta = 1

        for l in range(20, lam+2):
            best_delta = min(best_delta,PBinom_Mechansim(na,nb,ni, k-s, eps,s, l, mode)+ 1/(2**l) )
        delta += w * best_delta

    delta += (1-total_w)

    return delta

def MinhashGraphPBinom(na, nb, ni_list, k_list, eps_list,lam):

    for ni,k in itertools.product(ni_list,k_list):
        row = []
        for eps in eps_list:
            row.append(minhash_PM(na,nb,ni,k,eps,lam, 0))
        plt.plot(eps_list, row, label = "$n_I$ = {}, k = {}".format(ni, k))

        print('n_I = {}, k = {} completed'.format(ni, k))

    # plt.axis([0, max(eps_list)+1,])
    plt.xticks(np.arange(min(eps_list), max(eps_list) + 0.25, 0.25))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.ylabel(r'$\delta$',rotation=0, fontsize=18)
    plt.yscale('log')
    plt.yticks(np.array([10**(-1 * i) for i in range(4,13)]))
    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    MinhashGraphPBinom(1000000, 1000000, [250000,500000,750000], [1000,2000],[0.5+i*0.25 for i in range(0,11)],40)
    # MinhashGraphPBinom(1000000, 1000000, [250000, 500000], [1000, 2000], [0.5 + i * 1 for i in range(0, 3)], 40)
    # MinhashGraphPBinom(1000000, 1000000, [500000], [500], [0.5 + i * 0.25 for i in range(0, 1)], 40)

