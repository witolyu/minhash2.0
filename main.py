import math
import matplotlib.pyplot as plt

from scipy.stats import binom

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def chernoff_bound(k,n,p):
    # if k > n*p:
    #     return -1
    assert k <= n * p
    a = float(k)/n
    D_a_p = a * math.log(a/p, math.e) + (1-a) * math.log((1-a)/(1-p), math.e)
    return math.e**(-n*D_a_p)

def success_p(min_value, num_of_trials):
    # Return the success probability of Bernoulli trial capturing the probability that all non intersection items are
    # hashed to a greater value than minimum across intersected item.
    return (1-min_value) ** num_of_trials

def num_good_hashes(na, nb, ni, k, theta_list):
    # this return the number of good hashes given a list of theta:
    # (1) it hashed intersected input to minimum.
    # (2) it hashes to within range of [1/2-theta, 1/2+theta].

    theta_list.sort(reverse=True)

    k_good = k

    x_list = []
    y_list = []

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

def sensitivity(s,k,nb):
    # return the probability of sensitivity s, with k iteration and size of honest party's input set being nb.
    return binom.pmf(s,k,1.0/nb)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # n = 1000000
    # k = 500
    #
    # # intersection size/ union size
    # intersection_ratio = 1/2
    # theta = 1/4
    #
    # min_thre = 1 / 4000000
    # max_thre = 1 / 250000
    #
    # print(success_p(min_thre, 500000))
    # print(success_p(max_thre, 500000))
    #
    #
    # print(((1-min_thre)**1000000 - (1-max_thre)**1000000)*intersection_ratio)

    # theta_list = [1/2 - 1/2**i for i in range(2,11)]
    theta_list = [0.02 * i for i in range(1, 25)]

    # num_good_hashes(1000000, 1000000, 500000, 500, theta_list)

    # num_good_hashes(1000000, 1000000, 250000, 500, theta_list)

    # num_good_hashes(1000000, 1000000, 750000, 500, theta_list)

    print(chernoff_bound(246, 250, 1499999/1500000))

    r_values = list(range(78 + 1))

    dist = [binom.pmf(r, 78, 0.2) for r in r_values]

    print(list(enumerate(dist)))

    print(sensitivity(1, 5, 5))

    print([sensitivity(s, 500, 1000000) for s in range(50)])

    print("test")





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
