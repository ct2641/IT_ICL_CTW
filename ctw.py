import numpy as np
from tqdm import tqdm
import torch

class ct:
    # create aa context tree randomly according to Bayes context tree
    # with prior: spawn w.p. 1-beta, prediction Dir(prior, ..., prior)
    def __init__(self, M=2, D=1, beta=0.5, prior=0.5):
        self.M = M # M-ary
        self.D = D # maximal depth
        self.beta = beta # probabilty of stop branching
        self.prior = prior # prior of theta_s is Dir(prior, ..., prior)
        self.alphabet = [i for i in range(M)] # alphabet

        self.T = set() # tree, set of all tree nodes
        self.Lv = dict() # leaves s: theta_s
        queue = [()]
        while queue:
            new_queue = []
            for q in queue:
                self.T.add(q)
                if len(q) == self.D or np.random.rand() <= self.beta:
                    self.Lv[q] = np.random.dirichlet([self.prior] * M)
                    # Hard code mixture distribution
                    # self.Lv[q] = np.random.dirichlet([2., 0.5, 0.5])
                    # self.Lv[q] = np.random.dirichlet([0.5, 2.0, 0.5])
                else:
                    for a in self.alphabet:
                        new_queue.append(tuple([a]+list(q)))
            queue = new_queue
    
    def find_suffix(self, seq : list[int]) -> tuple[int]:
        # len(seq) >= D
        context = tuple(seq[-self.D:])
        while context not in self.Lv:
            context = context[1:]
        return context

    def sample(self, n : int, seq = None, verbose = False):
        if seq == None or len(seq) < self.D:
            seq = list(np.random.choice(self.alphabet, self.D))
        if verbose:
            print('generate sequence')
            print(f'length {n}')
            for _ in tqdm(range(n)):
                context = self.find_suffix(seq)
                seq += [np.random.choice(self.alphabet, p=self.Lv[context])]
        else:
            for _ in range(n):
                context = self.find_suffix(seq)
                seq += [np.random.choice(self.alphabet, p=self.Lv[context])]
        return seq
    
    def compress(self, seq : str, verbose : bool = False):
        """compute the entropy rate"""
        if len(seq) <= self.D:
            return 0
        ans = 0
        if verbose:
            print('Compress seq by CT')
            for key in self.Lv:
                print(f'{key:>6}:{self.Lv[key]}')
            for i in tqdm(range(self.D, len(seq))):
                context = self.find_suffix(seq[:i])
                ans -= np.log(self.Lv[context][int(seq[i])])
        else:
            for i in range(self.D, len(seq)):
                context = self.find_suffix(seq[:i])
                ans -= np.log(self.Lv[context][int(seq[i])])
        return ans / (len(seq) - self.D)


class ctw_model:
    def __init__(self, M=2, D=1, beta=0.5, prior=0.5, seq = None):
        self.M = M # M-ary
        self.D = D # maximal depth
        self.beta = beta # probabilty of stop branching
        self.prior = prior # prior of theta_s is Dir(prior, ..., prior)
        self.alphabet = [i for i in range(M)] # alphabet
        if seq == None or len(seq) < D:
            self.seq = [0] * D # observed seq, init with pseudo x_{-D}^{-1} = [0,0,...,0]
        else:
            self.seq = seq
        self.Tc = {(): np.array([self.prior] * M)} # prior of theta in complete tree of depth D
        self.log_pe = {(): 0.} # log(P_{e,s})
        self.log_pw = {(): 0.} # log(P_{w,s})
        self.cnt = {(): np.array([0] * M)} # count vectors n_s(a)
        self.tot = {(): 0} # total count of a context sum_a n_s(a)

        contexts = [()]
        for _ in range(D):
            contexts = [ tuple(list(a) + [b]) for a in contexts for b in self.alphabet]
            for c in contexts:
                self.Tc[c] = np.array([self.prior] * M)
                self.log_pe[c] = 0.
                self.log_pw[c] = 0.
                self.cnt[c] = np.array([0] * M)
                self.tot[c] = 0
        
    def get_tree(self):
        # return the tree with posterior of theta_s
        return self.Tc

    def update_seq(self, sym):
        # observe a new symbol sym
        # update auxiliary parameters for posterior
        # add sym to seq
        def update(seq, sym):
            for i in range(len(seq)+1):
                context = seq[i:]
                self.cnt[context][int(sym)] += 1
                self.tot[context] += 1
                self.log_pe[context] += np.log(self.cnt[context][int(sym)] + self.prior - 1) \
                                        - np.log(self.tot[context] + self.M * self.prior - 1)
                if len(context) == self.D:
                    self.log_pw[context] = self.log_pe[context]
                else:
                    tmp = sum([self.log_pw[ tuple([a]+list(context)) ] for a in self.alphabet]) # children Pw
                    tmp = tmp - self.log_pe[context] + np.log((1-self.beta) / self.beta)
                    self.log_pw[context] = self.log_pe[context] + np.log(self.beta)
                    self.log_pw[context] += tmp if tmp > 15 else np.log(1 + np.exp(tmp)) # trick to avoid overflow
        update(tuple(self.seq[-self.D:]), sym)
        self.seq += [sym]

    def predict(self):
        """return next token probabilty vector, weights over the path"""
        context = tuple(self.seq[-self.D:])
        lw = np.zeros(self.D + 1)
        for i in range(self.D):
            si1 = context[-i-1:]
            x, si = si1[0], si1[1:]
            tmp = 0
            for a in self.alphabet:
                tmp += self.log_pw[ tuple([a]+list(si)) ] if a != x else 0
            if i+1 == self.D:
                #lw[i+1] = lw[i] + np.log(1) \
                #                + self.log_pe[si1] + tmp - self.log_pe[si]
                lw[i+1] = lw[i] + np.log((1-self.beta) / self.beta) \
                                + self.log_pe[si1] + tmp - self.log_pe[si]
            else:
                lw[i+1] = lw[i] + np.log((1-self.beta)) \
                                + self.log_pe[si1] + tmp - self.log_pe[si]
        lw = lw - max(lw) # avoid overflow
        w = np.exp(lw)/sum(np.exp(lw))
        prob_vec = np.zeros(self.M)
        for i in range(len(w)):
            si = context[i:]
            prob_vec += w[self.D-i] * (self.Tc[si] + self.cnt[si]) / np.sum(self.Tc[si] + self.cnt[si])
        return prob_vec, w

def generate_seq(args):
    i, seed, M, D, beta, alpha, seq_len = args
    print(f'Generating context tree {i}')
    np.random.seed(seed + i)
    CT = ct(M, D, beta=beta, prior=alpha) # create a context tree
    seq = CT.sample(seq_len, verbose = False)
    seq = torch.tensor(seq)
    return seq
