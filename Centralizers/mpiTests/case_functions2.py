import numpy as np

def get_parameters(case: int,
                   min_power:int,
                   max_power:int,
                   min_coeff:int,
                   max_coeff:int,
                   ):
    cases={
        000: arbitrary,
        101: alpha_beta_zero,
        777: nonPropCase,
        888: propCase,
        1: case1,
        2: case2,
        3: case3,
        4: case4,
        5: case5,
        6: case6,
        7: case7,
        8: case8,
        9: case9

    }

    return cases[case](min_power, max_power, min_coeff, max_coeff)


def arbitrary(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(0, max_power + 1)
    k = np.random.randint(0, max_power + 1)
    n = np.random.randint(0, max_power + 1)
    m = np.random.randint(0, max_power + 1)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    # p = np.random.randint(0, 2)
    # if p == 0:
    #     alpha = 0
    # else:
    #     beta = 0

    return l, k, n, m, alpha, beta

def nonPropCase(min_power, max_power, min_coeff, max_coeff):
    while True:
        l = np.random.randint(0, max_power)
        k = l + 1
        n = np.random.randint(0, max_power)
        m = n + 1

        a = np.random.randint(min_coeff, max_coeff)
        alpha = -a * m
        beta = a * k

        if alpha == -beta:
            continue
        else:
            return l, k, n, m, alpha, beta

def propCase(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power)
    k = l + 1
    n = l
    m = n + 1

    a = np.random.randint(min_coeff, max_coeff)
    alpha = -a * m
    beta = a * k

    return l, k, n, m, alpha, beta

def alpha_beta_zero(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power + 1)
    m = np.random.randint(1, max_power + 1)
    m = 1

    k = np.random.randint(1, max_power + 1)
    n = np.random.randint(1, max_power + 1)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)
    alpha = 0



    return l, k, n, m, alpha, beta


def case1(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power + 1)
    k = l
    n = np.random.randint(1, max_power + 1)
    m = n

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    # p = np.random.randint(0, 2)
    # if p == 0:
    #     alpha = 0
    # else:
    #     beta = 0

    return l, k, n, m, alpha, beta

def case2(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(2, max_power + 1)
    k = l
    m = np.random.randint(2, max_power + 1)
    n = np.random.randint(0, m)
    n = 0

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    # p = np.random.randint(0, 2)
    # if p == 0:
    #     alpha = 0
    # else:
    #     beta = 0

    return l, k, n, m, alpha, beta


def case3(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power+1)
    k = l
    n = np.random.randint(1, max_power+1)
    m = np.random.randint(0, n)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta


def case4(min_power, max_power, min_coeff, max_coeff):
    k = np.random.randint(1, max_power+1)
    k = 2
    l = np.random.randint(0, k)
    l = k-2

    m = np.random.randint(1, max_power+1)
    m = k-1
    n = m

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta


def case5(min_power, max_power, min_coeff, max_coeff):
    k = np.random.randint(1, max_power+1)
    l = np.random.randint(0, k)

    m = np.random.randint(1, max_power+1)
    n = np.random.randint(0, m)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta

def case6(min_power, max_power, min_coeff, max_coeff):
    k = np.random.randint(2, max_power+1)
    l = np.random.randint(0, k)
    l = 0

    n = np.random.randint(1, max_power+1)
    m = np.random.randint(0, n)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta

def case7(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power+1)
    k = np.random.randint(0, l)

    m = np.random.randint(0, max_power+1)
    n = m

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta

def case8(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power+1)
    k = np.random.randint(0, l)

    m = np.random.randint(1, max_power+1)
    n = np.random.randint(0, m)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta

def case9(min_power, max_power, min_coeff, max_coeff):
    l = np.random.randint(1, max_power+1)
    k = np.random.randint(0, l)

    n = np.random.randint(1, max_power+1)
    m = np.random.randint(0, n)

    alpha = np.random.randint(min_coeff, max_coeff)
    beta = np.random.randint(min_coeff, max_coeff)

    return l, k, n, m, alpha, beta