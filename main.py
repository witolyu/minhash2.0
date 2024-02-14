#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import itertools
from scipy.stats import binom

# set true to represent delta in base 2, otherwise in base 10.
base2 = True
hockeyStickDivergence = False  # set true to use HSD instead of CDF tail bound.
lite = False


def Sensitivity(s, k, nb):
    # return the probability of sensitivity s, with k iteration and size of honest party's input set being nb.
    return binom.pmf(s, k, 1.0 / (nb + 1))


def SensitivityList(k, nb, lam):
    # return a list of tuples of (sensitivity and probability), only keep those with greater than 2^-lam probability
    res = []
    for s in range(0, k + 1):
        res.append((s, Sensitivity(s, k, nb)))
        if 1 - binom.cdf(s, k, 1.0 / (nb + 1)) < 2 ** (-(lam + 1)):
            break
    return res


def BinomChernoffBound(k, n, p):
    # if k > n*p:
    #     return -1
    assert k <= n * p
    a = float(k) / n
    D_a_p = a * math.log(a / p, math.e) + (1 - a) * math.log((1 - a) / (1 - p), math.e)
    return math.e ** (-n * D_a_p)


def NumGoodHashes(na, nb, ni, k, lam):
    # this return the number of good hashes given a list of theta:
    # (1) it hashed intersected input to minimum.
    # (2) it hashes to within range of [1/2-theta, 1/2+theta].

    res = []

    theta_list = [0.02 * i for i in range(24, 0, -1)]

    # k_good = k

    for theta in theta_list:
        # this is the success probability of the good hash event for each iteration.
        p = ((1 / 2 + theta) ** (na / (nb - ni)) - (1 / 2 - theta) ** (na / (nb - ni))) * ni / na

        # while True:
        #     if k_good == 0:
        #         break
        #     if k_good >= p * k:
        #         # As we want lower bound (with negligible probability), k_good should be less than expected.
        #         k_good -= 1
        #         continue
        #
        #     # print(theta,k_good,k,p)
        #     # tail = BinomChernoffBound(k_good,k,p)
        #     tail = binom.cdf(k_good, k, p)
        #
        #     if tail <= 2 ** (-lam):
        #         res.append((theta, k_good))
        #         break
        #     else:
        #         k_good -= 1

        k_good = binary_search_for_k_good(k,p,lam)
        res.append((theta, k_good))

        # print(theta, k_good,p)

    return res

def binary_search_for_k_good(k, p, lam):
    left = 0
    right = math.floor(k*p)
    while left <= right:
        mid = (left + right) // 2
        tail = binom.cdf(mid, k, p)
        if tail <= 2 ** (-lam):
            if mid == math.floor(k*p) or binom.cdf(mid + 1, k, p) > 2 ** (-lam):
                return mid
            else:
                left = mid + 1
        else:
            right = mid - 1
    return 0

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
        p = ((1 / 2 + theta) ** (na / (nb - ni)) - (1 / 2 - theta) ** (na / (nb - ni))) * ni / na

        while True:
            if k_good == 0:
                break
            if k_good >= p * k:
                k_good -= 1
                continue

            # print(theta,k_good,k,p)
            tail = BinomChernoffBound(k_good, k, p)

            if tail <= 2 ** (-40):
                break
            else:
                k_good -= 1

        # print(theta, k_good,p)
        x_list.append(theta)
        y_list.append(k_good)
    # print(x_list, y_list) # uncomment this to show the experiment process. 
    plt.plot(x_list, y_list, 'ro')
    plt.axis([0, 0.5, 0, 250])
    plt.xlabel('Theta')
    plt.ylabel('# of "good" hash functions')
    plt.suptitle('n_A = {}, n_B = {}, n_I = {}, k = {}'.format(na, nb, ni, k))
    # plt.show()
    filename = f"{na}_{nb}_{k}_NumGoodHashesGraph.png"
    plt.savefig(filename)
    plt.close()


# Poisson binomial mechanism, return delta for given epsilon and k.

def BinomMechanism(p, k, eps, s):
    e_eps = math.e ** eps
    e_eps_s = math.e ** (eps / s)
    q = 1 - p

    if hockeyStickDivergence:
        # FIXME: Implement Binary search and Hockey Stick Divergence
        l = math.ceil((k * p + s * p + s * q * e_eps_s) / (e_eps_s * q + p))
        l_delta = binom.cdf(l - 1, k, p)

        # right tail
        r = math.floor((k * p * e_eps_s) / (q + e_eps_s * p))
        r_delta = 1 - binom.cdf(r, k, p)

    else:
        # left tail, use a looser bound compared to the paper, without assume p < 1/2.
        l = math.ceil((k * p + s * p + s * q * e_eps_s) / (e_eps_s * q + p))
        l_delta = binom.cdf(l - 1, k, p)

        # right tail
        r = math.floor((k * p * e_eps_s) / (q + e_eps_s * p))
        r_delta = 1 - binom.cdf(r, k, p)

    return max(l_delta, r_delta)


def PBinomMechansim(na, nb, ni, k, eps, s, lam, mode):
    # mode = 0: throw away half; mode = 1: brute force; mode = 2: ratio between upper and lower bound.
    theta_k_list = NumGoodHashes(na, nb, ni, k, lam)
    min_delta = 1
    for theta, k_good in theta_k_list:
        k_good_half = math.floor(k_good / 2)
        p = 0.5 - theta
        delta = BinomMechanism(p, k_good_half, eps, s)
        min_delta = min(min_delta, delta)

    return min_delta

# def PBinomMechansim(n, J, k, eps, s, lam, mode):
#     ni = 2*n*J/(J+1)
#     na = n
#     nb = n
#     # mode = 0: throw away half; mode = 1: brute force; mode = 2: ratio between upper and lower bound.
#     theta_k_list = NumGoodHashes(na, nb, ni, k, lam)
#     min_delta = 1
#     for theta, k_good in theta_k_list:
#         k_good_half = math.floor(k_good / 2)
#         p = 0.5 - theta
#         delta = BinomMechanism(p, k_good_half, eps, s)
#         min_delta = min(min_delta, delta)
#
#     return min_delta

# This compute the "real delta", i.e., the weighted sum of delta for different sensitivity, as well as a failure rate.
def MinhashPM(na, nb, ni, k, eps, lam, mode):
    # calculate sensitivity

    delta = 0

    # total_w = 0

    s_list = SensitivityList(k, nb, lam)

    # optimize the failure probability to claim good hashes, no need to make this extremely small, when delta term is large
    # optimize the failure probability to claim good hashes, no need to make this extremely small, when delta term is large
    for s, w in s_list:
        # total_w += w

        if s == 0:
            continue

        best_delta = 1
        if lite:
            best_delta = min(best_delta, PBinomMechansim(na, nb, ni, k - s, eps, s, lam, mode) + 1 / (2 ** lam))
        else:
            for l in range(20, lam + 1, 2):
                best_delta = min(best_delta, PBinomMechansim(na, nb, ni, k - s, eps, s, l, mode) + 1 / (2 ** l))
        delta += w * best_delta

    delta += 2**(-(lam+1))

    return delta


# Compute dealta for Binomial Mechanism.
# This compute the "real delta", i.e., the weighted sum of delta for different sensitivity, as well as a failure rate.
def MinhashBM(na, nb, ni, k, eps, lam, mode):
    # Jaccard Similarity

    J = ni / (na + nb - ni)

    # calculate sensitivity

    delta = 0

    total_w = 0

    s_list = SensitivityList(k, nb, lam)
    # print(s_list) # uncomment this to show the experiment process. 

    # optimize the failure probability to claim good hashes, no need to make this extremely small, when delta term is large
    for s, w in s_list:
        total_w += w

        if s == 0:
            continue

        delta += w * BinomMechanism(J, k, eps, s)
        # print(delta) # uncomment this to show the experiment process. 
    delta += 2 ** (-(lam + 1))
    return delta


def MinhashGraphBinom(na, nb, ni_list, k_list, eps_list, lam):
    min_delta = 1
    max_delta = 0
    for ni, k in itertools.product(ni_list, k_list):
        row = []
        print('n_I = {}, k = {}'.format(ni, k))  # uncomment this to show the experiment process.
        for eps in eps_list:
            delta = MinhashBM(na, nb, ni, k, eps, lam, 0)
            min_delta = min(min_delta, delta)
            max_delta = max(max_delta, delta)
            row.append(delta)
            print("eps = {}, delta = {}".format(eps, delta))
        plt.plot(eps_list, row, label="$n_I$ = {}, k = {}".format(ni, k))



    # plt.axis([0, max(eps_list)+1,])
    plt.xticks(np.arange(min(eps_list), max(eps_list) + 0.25, 0.25))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.ylabel(r'$\delta$', rotation=0, fontsize=18)
    if base2:
        plt.yscale('log', base=2)
        plt.yticks(np.array([2 ** (-1 * i) for i in range(16, 64, 8)]))
    else:
        plt.yscale('log')
        plt.yticks(np.array([10 ** (-1 * i) for i in range(6, 13)]))
    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    # plt.show()
    filename = f"{na}_{nb}_MinhashGraphBinom_{lam}.png"
    plt.savefig(filename)
    plt.close()


def MinhashGraphPBinom(na, nb, ni_list, k_list, eps_list, lam):
    min_delta = 1
    max_delta = 0
    for ni, k in itertools.product(ni_list, k_list):
        print('Testing n_I = {}, k = {}'.format(ni, k))  # uncomment this to show the experiment process.
        row = []
        for eps in eps_list:
            delta = MinhashPM(na, nb, ni, k, eps, lam, 0)
            min_delta = min(min_delta,delta)
            max_delta = max(max_delta,delta)
            row.append(delta)
            print("eps = {}, delta = {}".format(eps,delta))
        plt.plot(eps_list, row, label="$n_I$ = {}, k = {}".format(ni, k))



    # plt.axis([0, max(eps_list)+1,])
    # plt.xticks(np.arange(min(eps_list), max(eps_list) + 0.25, 0.25))
    plt.xticks(np.array(eps_list))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.ylabel(r'$\delta$', rotation=0, fontsize=18)
    if base2:
        plt.yscale('log', base=2)
        # plt.yticks(np.array([2 ** (-1 * i) for i in range(16, 45, 4)]))
        plt.yticks(np.array([2 ** (-1 * i) for i in range(math.floor(-math.log(max_delta,2)),
                                                          math.ceil(-math.log(min_delta,2)), 4)]))
    else:
        plt.yscale('log')
        plt.yticks(np.array([10 ** (-1 * i) for i in range(6, 13)]))
    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    # plt.show()
    filename = f"{na}_{nb}_MinhashGraphPBinom_{lam}.png"
    plt.savefig(filename)
    plt.close()


def MinhashPMK(na, nb, J, eps, delta, lam, mode):
    # Given the jaccard index, compute the number of hashes needed to achieve set eps delta.

    # find the smallest K that satisfies eps, delta.
    ni = math.floor((na + nb) * J / (1 + J))

    precision = 100

    low = max(int(2500/ precision), 1)
    high = int(3000/ precision)

    # print("Testing K = {}".format(high * precision)) # uncomment this to show the experiment process.
    if MinhashPM(na, nb, ni, high * precision, eps, lam, mode) > delta:
        return np.nan
    # The binary search make contains small errors, as we may not be dealing with strictly decreasing function.
    # But it should be fine.
    while low < high:
        mid = (low + high) // 2
        # print("Testing K = {}".format(mid * precision)) # uncomment this to show the experiment process.
        if MinhashPM(na, nb, ni, mid * precision, eps, lam, mode) < delta:
            high = mid
        else:
            low = mid + 1

    assert low == high
    return high * precision


def MinhashGraphPBinomJaccardVsK(na, nb, eps_list, delta_list, lam):
    J_list = [0.05 * i for i in range(1, 20)]
    # J_list = [0.25 * i for i in range(1, 4)]
    # J_list = [0.1 * i for i in range(1, 10)]
    res = []

    for eps, delta in itertools.product(eps_list, delta_list):
        # print("Computing for eps = {} and delta = {}".format(eps,delta)) # uncomment this to show the experiment process. 
        row = []
        for J in J_list:
            row.append(MinhashPMK(na, nb, J, eps, delta, lam, 0))
            # print("J = {} completed.".format(J)) # uncomment this to show the experiment process. 
        plt.plot(J_list, row, 'o-', label=r"$\epsilon$ = %d $\delta$ = $2^{%d}$" % (eps, int(math.log(delta, 2))))
        res.append(row)
        # print('eps = {}, delta = {} completed'.format(eps, delta)) # uncomment this to show the experiment process. 

    # Since it is time-consuming, therefore print the output.
    # print(res)

    plt.xticks(np.arange(0.05, 1, 0.05))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'Jaccard Index', fontsize=18)
    plt.ylabel(r'Number of Iterations', fontsize=18)

    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    # plt.show()
    filename = f"{na}_{nb}_MinhashGraphPBinomJaccardVsK_{lam}.png"
    plt.savefig(filename)
    plt.close()


def MinhashBMK(na, nb, J, eps, delta, lam, mode):
    # Given the jaccard index, compute the number of hashes needed to achieve set eps delta.

    # find the smallest K that satisfies eps, delta.
    ni = math.floor((na + nb) * J / (1 + J))

    precision = 10

    low = max(int(10 / precision), 1)
    high = int(500 / precision)

    # print("Testing K = {}".format(high * precision)) # uncomment this to show the experiment process. 
    if MinhashBM(na, nb, ni, high * precision, eps, lam, mode) > delta:
        return np.nan
    # The binary search make contains small errors, as we may not be dealing with strictly decreasing function.
    # But it should be fine.
    while low < high:
        mid = (low + high) // 2
        # print("Testing K = {}".format(mid * precision)) # uncomment this to show the experiment process. 
        if MinhashBM(na, nb, ni, mid * precision, eps, lam, mode) < delta:
            high = mid
        else:
            low = mid + 1

    assert low == high
    return high * precision


def MinhashGraphBinomJaccardVsK(na, nb, eps_list, delta_list, lam):
    J_list = [0.05 * i for i in range(1, 20)]
    # J_list = [0.25 * i for i in range(1, 4)]
    # J_list = [0.1 * i for i in range(1, 10)]
    res = []

    for eps, delta in itertools.product(eps_list, delta_list):
        # print("Computing for eps = {} and delta = {}".format(eps,delta)) # uncomment this to show the experiment process. 
        row = []
        for J in J_list:
            row.append(MinhashBMK(na, nb, J, eps, delta, lam, 0))
            # print("J = {} completed.".format(J)) # uncomment this to show the experiment process. 
        plt.plot(J_list, row, 'o-', label=r"$\epsilon$ = %f $\delta$ = $2^{%d}$" % (eps, int(math.log(delta, 2))))
        res.append(row)
        # print('eps = {}, delta = {} completed'.format(eps, delta)) # uncomment this to show the experiment process. 

    # Since it is time-consuming, therefore print the output.
    # print(res)

    plt.xticks(np.arange(0.05, 1, 0.05))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'Jaccard Index', fontsize=18)
    plt.ylabel(r'Number of Iterations', fontsize=18)

    # plt.suptitle('n_A = {}, n_B = {}, n_I = {}'.format(na, nb, ni), fontsize=14)
    plt.legend()
    # plt.show()
    filename = f"{na}_{nb}_MinhashGraphBinomJaccardVsK_{lam}.png"
    plt.savefig(filename)
    plt.close()


def MinhashPBinomJaccardVsAccuracyCompareWithSFM(n, eps_list, delta_list, lam):
    # J_list = [0.05 * i for i in range(1, 20)]
    # J_list = [0.05,0.1,0.15,0.85,0.9,0.95]
    J_list = [0.7]

    res = []

    for eps, delta in itertools.product(eps_list, delta_list):
        # print("Computing for eps = {} and delta = {}".format(eps,delta)) # uncomment this to show the experiment process.
        row = []

        acc_SFM = []
        acc_MH = []

        for JI in J_list:
            # computing the number of iteration and sketch size

            k = MinhashPMK(n, n, JI, eps, delta, lam, 0)

            if math.isnan(k):
                print("cannot find a good k for JI={}",JI)
            else:

                # k = math.ceil(MinhashPMK(n, n, JI, eps, delta, lam, 0))

                sketchsize = k * 64

                # computing the empirical error for minhash

                n_U = 2 * n / (JI + 1)

                num_samples = 1000

                # Sampling from the binomial distribution
                samples_matching = np.random.binomial(k, JI, num_samples)

                samples_JI = [val / k for val in samples_matching]

                # RRMSE_JI = (np.mean([(val - JI) ** 2 for val in samples_JI])) ** (1 / 2) / JI

                samples_Union = [2 * n / (val / k + 1) for val in samples_matching]

                union_se = (np.mean([(val - n_U) ** 2 for val in samples_Union])) ** (1 / 2)

                RRMSE_U_MH = union_se / n_U

                acc_MH.append(RRMSE_U_MH)

                # computing the estimated error for SFM

                RRMSE_U_SFM = SFM_accuracy(n,JI, eps, sketchsize)

                acc_SFM.append(SFM_accuracy(n,JI, eps, sketchsize))

                print(JI,n_U,k, sketchsize,RRMSE_U_MH,RRMSE_U_SFM)

        # print(acc_SFM)
        # print(acc_MH)
        # print('eps = {}, delta = {} completed'.format(eps, delta)) # uncomment this to show the experiment process.

    # Since it is time-consuming, therefore print the output.
    # print(res)

def SFM_accuracy(n,JI, eps, sketchsize):
    # p = math.e ** eps / (math.e ** eps + 1)
    # q = 1 - p

    # I assume this need to be changed for different n?
    P = 24

    B = sketchsize/P

    n_U = 2*n/(JI+1)
    rho = [2 ** (-min(j, P - 1)) / B for j in range(P)]
    gamma = [1 - rho[j] for j in range(P)]

    # given the orginal eps, derive the equivalence eps_star after the (union) merge, according to Theorem 4.8

    eps_star = - math.log(2 * math.e ** (-eps) - math.e ** (-2 * eps))

    # Compute the corresponding p and q, according to Definition 4.6
    p = math.e ** eps_star / (math.e ** eps_star + 1)
    q = 1 - p

    # derive the SE for cardinality estimation through log likelihood,
    # according to Sec 5.1, equation 2. (squareroot of inverse of the hessian)

    estimated_SE = (B * (p - q) * sum([(math.log(gamma[j])) ** 2 * gamma[j] ** n_U *
                                       (p / (p - (p - q) * gamma[j] ** n_U) - (1 - p) / (
                                                   1 - p + (p - q) * gamma[j] ** n_U)) for j in range(P)])) ** (-1 / 2)

    return estimated_SE/n_U


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 1000000
    MinhashPBinomJaccardVsAccuracyCompareWithSFM(n, eps_list=[2], delta_list=[2**-40], lam=45)
    # k=2000
    # for J in [0.1,0.3,0.5,0.7,0.9]:
    #     print("J={}".format(J))
    #     ni = 2*n*J/(J+1)
    #     for eps in [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]:
    #         print(MinhashPM(n, n, ni, k, eps, 40, 0))

    # print(SFM_accuracy(n,0.5, 1, 620800))
    # print(MinhashPM(n, n, 666666, 10000, 1, 40, 0))


    # n = 10000
    # intersection_size_list = [2/11*n, 0.3*n, 2/3*n, 0.7*n,18/19*n]
    # hash_num_list = [2000]
    # epsilon_list = [2]
    # delta_list = [10 ** (-5), 10 ** (-6)]
    #
    # # lambda should be set to be smaller than the lg2 of the last of the delta list.
    # # This capture the failure probability when the sensitivity is too high.
    # lam = 40
    #
    # print("Now testing public hash setting with individual high min-entropy, n={}".format(n))
    # print("eps vs delta:")
    # MinhashGraphPBinom(n, n, intersection_size_list, hash_num_list, epsilon_list, lam)
    # print("Finished!")
    # print("Iterations vs Jaccard: This will take a while...")
    # MinhashGraphPBinomJaccardVsK(n,n, epsilon_list, delta_list, lam)
    # print("Finished!")
    #
    # n = 1000
    # intersection_size_list = [2 / 11 * n, 0.3 * n, 2 / 3 * n, 0.7 * n, 18 / 19 * n]
    # hash_num_list = [200]
    # epsilon_list = [2]
    # delta_list = [10 ** (-5), 10 ** (-6)]
    #
    # # lambda should be set to be smaller than the lg2 of the last of the delta list.
    # # This capture the failure probability when the sensitivity is too high.
    # lam = 40
    #
    # print("Now testing curator setting, n={}".format(n))
    # print("eps vs delta:")
    # MinhashGraphBinom(n, n, intersection_size_list, hash_num_list, epsilon_list, lam)
    # print("Finished!")
    # print("Iterations vs Jaccard:")
    # MinhashGraphBinomJaccardVsK(n,n, epsilon_list, delta_list, lam)
    # print("Finished!")

    # MinhashGraphBinomJaccardVsK(100000, 100000,  [1,2], [2**(-30),2**(-40),2**(-50)], 60)  # n = 100k