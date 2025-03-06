import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import NMF

from copy import copy
from scipy.sparse import csr_matrix

from GraphCreation import *


def tij2tijtau(df, n_jobs = 8, verbose = 4):
    '''Converts a pandas dataframe from the format (i, j, t) to the format (i, j, t, τ)'''
    
    # find all the possible interacting pairs
    df = df.set_index(['i', 'j'])
    df = df.sort_index()
    all_indeces = df.index.unique()

    # run the algorithm in parallel for each pair
    Pl = Parallel(n_jobs = n_jobs, verbose = verbose)
    result = Pl(delayed(tij2tijtauIndex)(df, index) for index in all_indeces)

    dfttau = pd.concat(result)
    dfttau = dfttau.rename(columns = {'t0': 't'})
    dfttau.t = (dfttau.t).astype(int)

    return dfttau


def tij2tijtauIndex(df, index):
    '''Converts a pandas dataframe from the format (i, j, t) to (i, j, t, τ) for a specific pair (i, j)'''
    
    # select the contacts related to a given pair
    ddf = df.loc[index].reset_index()
    
    # find contiguous events
    ddf['diff'] = ddf.t - np.roll(ddf.t,1) - 1
    
    # attribute to all them the same initial time
    v = np.zeros(len(ddf))
    idx = ddf['diff'] != 0
    v[idx] = ddf[idx].t
    ddf['t0'] = v
    ddf.t0 = ddf.t0.replace(0, method='ffill').values
    ddf = ddf[['t', 'i', 'j', 't0']]
    
    # compute the contact duration
    ddf['τ'] = np.ones(len(ddf))
    ddf = ddf.groupby(['t0', 'i', 'j']).sum().reset_index()[['t0', 'i', 'j', 'τ']]
    
    return ddf


def NMF_kmeans(M,k):

    n, _ = M.shape
    Mt = M + np.eye(n)*np.mean(M[M.nonzero()])
    Mt = Mt/np.mean(Mt)
    Y = NMF(n_components = k).fit(Mt).components_
    kmeans = KMeans(n_clusters = k, n_init = 10).fit(Y.T)
    return kmeans.labels_, np.abs(kmeans.score(Y.T))


def ClusterNMF(M, k):
    
    est_ℓ, score = NMF_kmeans(M, k)
    
    for i in range(20):
        est_ℓ_, score_ = NMF_kmeans(M, k)
        if score_ < score:
            score = score_
            est_ℓ = est_ℓ_
            
    return est_ℓ


def MakeTemporal(dft, timeFR, time_aggregation=10):
    '''This function takes an edge list (dft) in the form of a pandas dataframe and all the time series stored in
    timeFR and creates a temporal graph
    
    Input:
        * dft (DataFrame): static graph with columns (i,j)
        * timeFR (list): list of timesequences (each timesequence is a list of timestamps)
        * time_aggragation (float): the desired time aggregation (in minutes)
    '''

    idx1 = []
    idx2 = []
    tv = []

    # map each edge to one of the edges from time_FR
    for x in range(len(dft)):
        i, j = dft.iloc[x].values
        q = np.random.randint(len(timeFR))

        for t in timeFR[q]:
            idx1.append(i)
            idx2.append(j)
            tv.append(t)

    # build the temporal dataframe
    DFT = pd.DataFrame(np.array([idx1, idx2, tv]).T, columns = ['i', 'j', 't'])
    DFT.t -= DFT.t.min()
    DFT.t /= time_aggregation*60 # aggregation
    DFT.t = DFT.t.astype(int)
    DFT['τ'] = 1
    DFT = DFT.groupby(['i', 'j', 't']).sum().reset_index()
    
    return DFT

def CosSim(x,y):
    '''Cosine similarity between two vectors'''
    return x@y/np.sqrt(x@x*y@y)



class SIR_model():
    '''
    Class that simulates an epidemics on temporal graph, using a SIR model.

    Input:
        * df (Dataframe): temporal graph in the form (i,j,t,τ) (NOTE: τ is the weight of the edge)
    '''
    def __init__(self, df):
        
        self.df = df
        self.nodes = np.unique(df[['i','j']].values)
        node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.df['i_index'] = self.df['i'].map(node_to_index)
        self.df['j_index'] = self.df['j'].map(node_to_index)

    
    def initialize_state(self, f):
        '''This function initilizes the states of an SIR model setting all nodes to the susceptible state and the seeds to the infected one
        
        Use: state = initialize_state(seeds, nodes)
        
        Inputs:
            - f (float): fraction of infected nodes

        '''
        
        state = dict()
        state['S'] = set(self.nodes)
        state['I'] = set(np.random.choice(self.nodes, int(f*len(self.nodes)), replace=False))
        
        for seed in state['I']:
            state['S'].remove(seed)
        state['R'] = set()

        return state
    

    def simulate(self, beta, mu,f):
        '''This function runs a SIR model on a temporal graph
                
        Inputs: 
            - beta (float): infection rate
            - mu (float): recovery rate
            - f (float): fraction of infected nodes
            
        Output:
            - state_ev (list of dictionaries): each element corresponds to the state at a given time step
        '''
        state = self.initialize_state(f)
        
        state_ev = [state]
        t_list = np.sort(self.df.t.unique())
        
        for t in np.sort(t_list):
            sub_graph = self.df[self.df.t == t]

            A = csr_matrix((sub_graph['τ'].to_numpy(), (sub_graph['i_index'], sub_graph['j_index'])), shape=(len(self.nodes), len(self.nodes)))

            state = self._update_step(state, A, beta, mu)
            state_ev.append(state)
            
            if (len(state['I'])==0): #or (len(state['S'])==0): # all infected have recovered or no more susceptibles
                break
        return state_ev
    
    
    def _update_step(self, state, A, beta, mu):
        '''Perform a single step in the SEIR simulation'''

        state_ = copy(state)

        state_ = self._update_state(state_, A, mu, 'I', 'R')
        state_ = self._update_state(state_, A, beta, 'S', 'I')

        return state_


    def _update_state(self, state, A, prob, value_old, value_new):
        '''Generate the transition between two states'''

        state_ = copy(state)

        if value_old == 'S':
            idxS = np.array([i in state_['S'] for i in self.nodes])
            I = np.array([1 if i in state_['I'] else 0 for i in self.nodes])
            pinf = 1-(1-prob)**(A[idxS]@I)
            idx = np.random.binomial(1, pinf) == 1

        else:
            idx = np.random.binomial(1, prob, len(state[value_old])) == 1
        
        # nodes that will change their state from value_old to value_new
        selected_nodes = [v for i, v in enumerate(state[value_old]) if idx[i]]

        # update
        state_[value_new] = state_[value_new].union(selected_nodes)
        state_[value_old] = state_[value_old].difference(selected_nodes)

        return state_




def synthetic_graphs_generation(path, n_graphs_per_type=250, n0=1000, γ = 0.8, time_aggregation = 10):
    """
    Function that generates n_graphs_per_type DCSBM, ER, Configuration and Geometric graphs in a temporal version. 
    The temporal sequences for each edge are taken by a real temporal network obtained through path.
    n0 is the mean number of nodes per graphs and  γ is its variance.
    It return the list of graphs dataframes in the format i,j,t.
    """


    # we build df_SP so that it contains (i,j,t, day)
    df_SP = pd.read_csv(path, header = None, sep = ' ', names = ['t', 'i', 'j','c_i','c_j'])
    df_SP = df_SP[['t','i','j']]
    df_SP['day'] = pd.to_datetime(df_SP.t, unit = 's')
    df_SP['day'] = df_SP.day.dt.day
    all_pairs = df_SP.groupby(['i', 'j', 'day']).size().reset_index()[['i', 'j', 'day']].values
    df_SP.set_index(['i', 'j', 'day'], inplace = True) # <-- so this is a time series of t, where i, j and day are the indeces

    timeF = []

    # a pair is in the form (i, j, day) and timeF stores all the interactions between i and j on day t
    for x, pair in enumerate(all_pairs):
        print(str(np.round((x+1)/len(all_pairs)*100,2)) + ' %', end = '\r')
        i, j, day = pair
        timeF.append(df_SP.loc[i, j, day].t.values) 
        
    # we then construct time FR that for each (i,j,t) stores the edge time-line, centered to 00:00 of the corresponding day 
    timeFR = [[] for x in range(len(timeF))]

    for a, T in enumerate(timeF):
        print('Time sequences extraction: '+str(np.round((a+1)/len(all_pairs)*100,2)) + ' %', end = '\r')
        for t in T:
            if (pd.to_datetime(t, unit = 's') - pd.to_datetime('1970-01-01 08:00:00')).days == 1:
                tt = (pd.to_datetime(t, unit = 's') - pd.to_datetime('1970-01-02 08:00:00')).total_seconds()
                
            else:
                tt = (pd.to_datetime(t, unit = 's') - pd.to_datetime('1970-01-01 08:00:00')).total_seconds()
                
            timeFR[a].append(tt)
    print('\n')

    # We generate a sequence of temporal graphs with n0 = 1000 as the average number of nodes and store them in the 
    # list DFT. For each type we generate 250 graphs

    DFT = []

    ### DCSBM
    k = 5
    c_in = 25
    c_out = 1
    C = np.ones((k,k))*c_out
    C += np.diag(np.ones(k))*(c_in - c_out)
    c = (c_in + (k-1)*c_out)/k
    symmetric = True
    make_connected = True



    for i in range(n_graphs_per_type):
        print('DCSBM: '+str(100*(i+1)/n_graphs_per_type) + ' %' , end = '\r')
        
        n = np.random.randint(int(n0*(1-γ)), int(n0*(1+γ)))
        θ = np.ones(n)
        ℓ = np.zeros(n)
        for i in range(k):
            ℓ[i*int(n/k): (i+1)*int(n/k)] = i
        ℓ = ℓ.astype(int)

        args = (C, c, ℓ, θ, symmetric, make_connected)
        dft = DCSBM(args)
        DFT.append(MakeTemporal(dft, timeFR,time_aggregation))
    print('\n')
        
    ### ER
    k = 1
    C = np.ones((k,k))*c

    for i in range(n_graphs_per_type):
        print('ER: '+str(100*(i+1)/n_graphs_per_type) + ' %' , end = '\r')
        
        n = np.random.randint(int(n0*(1-γ)), int(n0*(1+γ)))
        ℓ = np.zeros(n).astype(int)
        θ = np.ones(n)
        args = (C, c, ℓ, θ, symmetric, make_connected)

        dft = DCSBM(args)
        DFT.append(MakeTemporal(dft, timeFR,time_aggregation))
    print('\n')

        
    ### Configuration model
    for i in range(n_graphs_per_type):
        print('Configuraion Model: '+str(100*(i+1)/n_graphs_per_type) + ' %' , end = '\r')
        
        n = np.random.randint(int(n0*(1-γ)), int(n0*(1+γ)))
        θ = np.random.uniform(3,10, n)**4
        θ = θ/np.mean(θ)
        ℓ = np.zeros(n).astype(int)
        args = (C, c, ℓ, θ, symmetric, make_connected)

        
        dft = DCSBM(args)
        DFT.append(MakeTemporal(dft, timeFR,time_aggregation))
    print('\n')

    ### Geometric model
    β = 20

    for i in range(n_graphs_per_type):
        print('Geometric Model: '+str(100*(i+1)/n_graphs_per_type) + ' %' , end = '\r')
        
        n = np.random.randint(int(n0*(1-γ)), int(n0*(1+γ)))
        
        r = np.random.uniform(0, 1, n)
        θ = np.random.uniform(0, 2*np.pi, n)
        X = np.zeros((n, 2))
        X[:,0] = r*np.cos(θ)
        X[:,1] = r*np.sin(θ)
        args = (X, c, β)

        
        dft = GeometricModel(args)
        DFT.append(MakeTemporal(dft, timeFR,time_aggregation))
        
    return DFT