#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:28:47 2018
This module contains three classes which are intended to represent complete
experiments that might be performed on atmospheric data. These are:
    
    DataProcessing:
        This class takes in an iris cube of data and allows various types of 
        processing to be performed on it.
        
    ClusteringExperiment:
        Takes a list of iris cubes each of which should be temporally continuous.
        These cubes can then undergo some dimensionality reduction, subsampling
        and filtering before being clustered using either a HMM or K means algorithm.
       
    MC_Cluster_Experiment:
        A wrapper class for ClusteringExperiment, which also allows the
        clustering of synthetic datasets,and resampling in order to perform
        tests of statistical significance and classifiability.
        
@author: dorrington
"""

import numpy as np
import iris
import itertools
    
###############################################################################
####SOME MISCELLANEOUS FUNCTIONS###############################################
###############################################################################
#This returns a matrix [KxK] of the RMS distance between the rows 
#of two matrices, shape [KxD]
def RMSE_matrix(C1,C2):
   K,D=C1.shape
   if C2.shape != (K,D):
       print("Warning: Trying to compare centroids of different cluster number or state space")
       return None
   else:
       #Uses broacasting to compute the L2 Distance between every
       #row vector in C1 and every row vector in C2
       A=np.sqrt(np.sum((C1[:,np.newaxis,:]-C2)**2,axis=2))
       return A
            

def get_cluster_cube(input_cube,states,as_anomaly=True):
    Ks=np.unique(states)
    if len(states)!=len(input_cube.coord('time').points):
        print(f"length of state vector was {len(states)}, was expecting {len(input_cube.coord('time').points)}")
        return None
    Lats=len(input_cube.coord('latitude').points)
    Lons=len(input_cube.coord('longitude').points)
    clusters=np.zeros([len(Ks),Lats,Lons])
    for i,K in enumerate(Ks):
        clusters[i] = input_cube.data[states==K].mean(axis=0)
    if as_anomaly:
        mean=input_cube.data.mean(axis=0)
        clusters-=mean[None,:,:]
    latitude,longitude=input_cube.dim_coords[1:]
    cluster_cube=iris.cube.Cube(data=clusters)
    cluster_cube.add_dim_coord(iris.coords.DimCoord(Ks,long_name="mean cluster composites"),0)
    cluster_cube.add_dim_coord(latitude,1)
    cluster_cube.add_dim_coord(longitude,2)
    return cluster_cube

###############################################################################
##########################EXPERIMENT CLASSES###################################
###############################################################################
"""
DataProcessing

Inputs: input_data - a data cube to be processed, at the moment only
        [time,lat,lon] is supported.

Methods: take_anomaly - subtracts the time mean from every gridpoint leaving
         anomalies
                        
         normalise - sets the (mean,variance) of every gridpoint to 0
         
         seasonally_detrend(Nmodes) - fits Nmodes sine functions to the data
         at each gridpoint with frequencies N yr^-1 for N in {1,...,Nmodes}
         and subtracts to create seasonally detrended data
         
         subset_months(subset) - Extracts the months in the specified subset.
         This should be either a season or an array of month labels ["Jan",...]
         
         subset_space(latmin,latmax,lonmin,lonmax) - keeps only gridpoints 
         within these bounds
         
         smooth(window,days) - applies a Laczos lowpass filter to the data, 
         removing timescales beneath 'days' days and with a 'window' point window
         
         EOF_projection - Project the data onto PCs and optionally store the EOFs
         
Attributes:
    
        raw_data - The original input data (not a copy, a pointer)
        
        data - The copy of raw_data subject to modification by the class' methods
        
        PCs - The principal components of self.data if calculated
        
        EOFs - The EOFs of self.data if calculated
        
        history - A record of every calculation done on the data in order.
         
"""
class DataProcessing:
    import iris
    def __init__(self,input_data=None):
        self.raw_data=input_data
        self.data=input_data
        self.PCs=None
        self.EOFs=None
        
        self.history=[]
                
    #subtract mean of every gridpoint
    def take_anomaly(self):
        time_mean=self.data.collapsed('time', iris.analysis.MEAN)
        anomaly=self.data-time_mean
        self.data=anomaly
        self.history.append("Each gridpoint's time mean subtracted.")
    #set (mean,variance) of every gridpoint to (0,1)
    def normalise(self):
        time_mean=self.data.collapsed('time', iris.analysis.MEAN)
        time_std=self.data.collapsed('time', iris.analysis.STD_DEV)
        normalised=(self.data-time_mean)/time_std
        self.data=normalised
        self.history.append("Each gridpoint normalised")
    def seasonally_detrend(self,Nmodes=4):
        from scipy.optimize import leastsq
        #intelligently guesses p
        def guess_p(N,tstd):
            p=np.zeros(2*(N+1))
            for i in range(0,N):
                p[2+2*i]=tstd/(i+1.0)
            return p        
        #defines multimode sine function for fitting
        def peval(x,p,N):
            ans=p[0]*x+p[1]
            for i in range(0,N):
                ans+=p[2*i+2] * np.sin(2 * np.pi * (i+1)/365.25 * x + p[2*i+3])
            return ans
        #defines error function for optimisation
        def residuals(p,y,x,N):
            return y - peval(x,p,N)
            
        N=Nmodes #The number of sine modes to fit with
        t=self.data.coord("time").points
        t=((t-t[0])/24) #makes hourly time units into daily
        t=t%365.25 #transforms series into true astronomical years
        
        detrended=np.zeros_like(self.data.data)
        for i in range(self.data.shape[1]):
            for j in range(self.data.shape[2]):
                griddata=self.data.data[:,i,j]
                tstd=griddata.std()
                p0=guess_p(N,tstd)
                plsq=leastsq(residuals,p0,args=(griddata,t,N))
                detrended[:,i,j]=self.data.data[:,i,j]-peval(t,plsq[0],N)
        self.data.data=detrended
        self.history.append(f"seasonally detrended with {Nmodes} modes.")
        
    def subset_months(self,subset):
        
        if   subset == "Spring" : months= ["Mar","Apr","May"]
        elif subset == "Summer" : months= ["Jun","Jul","Aug"]
        elif subset == "Autumn" : months= ["Sep","Oct","Nov"]
        elif subset == "Winter" : months= ["Dec","Jan","Feb"]
        elif subset == "Winter_long" : months=["Dec","Jan","Feb","Mar"]
        else: months = subset
        if self.data.coord("month")[0].shape !=1:
            print("Warning: No unique month label for each day")
        self.data=self.data.extract(iris.Constraint(month=months))
        self.history.append(f"subsetted by month to {months}.")
        
    def subset_space(self,latmin,latmax,lonmin,lonmax):
        self.data=self.data.intersection(longitude=[lonmin,lonmax],latitude=[latmin,latmax])
        self.history.append(f"subsetted to Lat [{latmin},{latmax}] Lon [{lonmin},{lonmax}].")
        
    def smooth(self,window=15,days=5.):
        
        def low_pass_weights(window, cutoff):
            order = ((window - 1) // 2 ) + 1
            nwts = 2 * order + 1
            w = np.zeros([nwts])
            n = nwts // 2
            w[n] = 2 * cutoff
            k = np.arange(1., n)
            sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
            firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
            w[n-1:0:-1] = firstfactor * sigma
            w[n+1:-1] = firstfactor * sigma
            return w[1:-1]
        
        weights=low_pass_weights(window, 1./days)
        smoothed=self.data.rolling_window("time",iris.analysis.SUM,len(weights),weights=weights)
        self.data=smoothed
        self.history.append(f"{days}-day lowpass Laczos filter with {window}-point window applied.")
        
    def EOF_projection(self,nEOFs,store_EOFs=True):
        from eofs.iris import Eof

        weighting_type='coslat' #this weighting ensures the variance of each gridpoint is area weighted    
        solver=Eof(self.data,weights=weighting_type) 
        self.PCs=solver.pcs(npcs=nEOFs)
        self.history.append(f"Leading {nEOFs} PCs calculated.")
        if store_EOFs:
            self.EOFs=solver.eofs(eofscaling=1,neofs=nEOFs)
            self.history.append(f"Leading {nEOFs} EOFs calculated.")
            

"""
ClusteringExperiment

Inputs: PC_data - a list of iris cubes storing principal components. Each cube
        in the list will be treated as being temporally continuous, representing
        ,for example, an entire season
        
        PCnum - The number of principal components to use for clustering
        
        K - The number of clusters to which the data is fitted.
        
        alg - A string specifying the clustering algorithm to use; either
        'Kmeans' or 'HMM'.
        
        alpha - An optional parameter which keeps only the fraction 'alpha'
        of slowest evolving points for clustering. For HMM the filtering is 
        only applied at the post processing step due to the need for evenly
        spaced data.
        
        sbsp - An optional subsampling to apply to the data before clustering
        
        numpy_input - A flag to enable numpy input instead of iris cube input.
        Not Extensively Tested.

Methods: setup_like(Experiment) - Initialises the experiment with the same
        flags, data and settings as the input Experiment.
           
         preprocess_data() - Performs all preprocessing on data prior to
         clustering such as dimensionality reduction, subsampling and any
         alpha filtering.

         cluster() - Performs the cluster analysis on the preprocessed data

         postprocess_data() - Performs any additional processing on the data
         and on the state vectors after clustering is complete
         
         get_centroids() - Returns the centre of the clusters in the state space
         used for clustering.
         
         get_var_ratios() - Calculates the ratio of the variance between
         cluster centroids to the occupation weighted intracluster variance.
         This is a common metric for the quality of the clustering fit to the data.
         
         _compute_var_ratio(data,states) - An internal method used by 
         get_var_ratios.
         
         get_significance(nsynth) - Computes the variance ratio for clustering
         on nsynth synthetic datasets of linear stochastic noise with the same
         autocorrelation structure as the real data. This allows it to compute
         the significance of the var_ratio found, with errors estimated by
         bootstrapping.

         get_details() - Prints a variety of information about settings and
         clustering results to terminal
         
         OLD BUT UNDER REVIEW:
             
         transmat - convenience function that computes the transition matrix
         from the state vector
         
         plot_clusters - convenience function that plots the cluster composites
         based on the unprocessed initial data.
         
Attributes:
    
        data - The original input data (not a copy, a pointer)
                
        PCnum - The number of PCs being used
        
        K - The number of clusters
        
        alg - The chosen algorithm
        
        alpha - The value of alpha being used
        
        subsample_rate - the subsample rate being used
        
        np - A flag for if numpy input data is being used
        
        preprocessed_data - The data produced by preprocess_data(), and
        which goes into the cluster() function
        
        quasistationary - A list oftruth values that state whether a data
        point falls below the alpha cutoff for slowly evolving datapoints
        
        states - A list of state vectors fitted to the preprocessed data
        
        postprocessed_data - The output of  postprocess_data()
        
        experiment_progress - A flag that increments as the methods of the
        class are called, keeping track of progress. Should be 3 at the end
        of the clustering experiment.         
"""         
class ClusteringExperiment:
        import numpy as np
        
        def __init__(self,PC_data=None,PCnum=None,K=None,alg=None,alpha=None,sbsp=None,numpy_input=False):
            self.data=PC_data
            self.PCnum=PCnum
            self.K=K
            self.alg=alg
            self.alpha=alpha
            self.subsample_rate=sbsp
            self.np=numpy_input
            
            self.preprocessed_data=None
            self.quasistationary=None
            self.states=None
            self.postprocessed_data=None
            self.experiment_progress=0
            
        #copies configuration from another experiment.
        def setup_like(self,Experiment):
            self.data=Experiment.data
            self.PCnum=Experiment.PCnum
            self.K=Experiment.K
            self.alg=Experiment.alg
            self.alpha=Experiment.alpha
            self.subsample_rate=Experiment.subsample_rate
          
        #summarises the experiment configuration and progress
        def get_details(self,long=False):
            print(f"{self.alg} algorithm, K={self.K} in {self.PCnum} PCs")
            print(f"alpha is {self.alpha}, subsampling rate is {self.subsample_rate}")
            if long:
                print(f"data is shape {self.data.shape}")
                if self.experiment_progress==0:
                    print(f"Experiment initialised")
                if self.experiment_progress==1:
                    print(f"Preprocessing performed")
                if self.experiment_progress==2:
                    print(f"Clustering performed")
                if self.experiment_progress==3:
                    print(f"Postprocessing performed")
                
            
        #Preprocess data for clustering. Minimally, extracts a numpy array
        #from the iris datacube if numpy_input is False
        def preprocess_data(self):
            
            #extract numpy data from cube
            if not self.np:
                self.preprocessed_data=[np.array(cube.data) for cube in self.data]
            else:
                self.preprocessed_data=np.array(self.data)
            if np.ndim(self.preprocessed_data[0])!=2:
                print("Expected data to be rank 2")
            
            #keeps only leading PCs
            if self.PCnum is not None:
                self.preprocessed_data=[series[:,:self.PCnum] for series in self.preprocessed_data]
                
            #Subsamples data
            if self.subsample_rate is not None:
                self.preprocessed_data=[series[::self.subsample_rate] for series in self.preprocessed_data]
            
            #filters out non-quasistaionary data
            if self.alpha is not None:
                if (self.alg != "HMM"):
                    self.alpha_filter()
                
            self.experiment_progress=1
    
            
        #applies the quasistaionary filter to the selected data
        def alpha_filter(self,data="pre"):
            
            phase_vel=[np.sqrt(np.sum((d[2:]-d[:-2])**2,axis=1)) for d in self.preprocessed_data]
            phase_vel=[np.array((v.mean(),*v,v.mean())) for v in phase_vel]    
            v_cutoff=np.percentile(np.concatenate(phase_vel).ravel(),100*self.alpha)
            self.quasistationary=[v<v_cutoff for v in phase_vel]
            
            if data=="pre":
                self.preprocessed_data=[d[q] for (d,q) in zip(self.preprocessed_data,self.quasistationary)]
            if data=="post":
                self.postprocessed_data=[d[q] for (d,q) in zip(self.postprocessed_data,self.quasistationary)]
        
        #Take preprocessed_data and cluster it according to the specified algorithm
        #saving the generated state vector.
        def cluster(self):
            if self.preprocessed_data is None:
                print("No preprocessed_data attribute found")
                return -1
            
            if self.alg=="Kmeans":
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=self.K,precompute_distances=True)
                km.fit(np.concatenate(self.preprocessed_data)) #flattens all dates together
                self.states=[km.predict(d) for d in self.preprocessed_data]
                
            elif self.alg=="HMM":
                from pomegranate import HiddenMarkovModel, MultivariateGaussianDistribution
                distribution=MultivariateGaussianDistribution
                hmm=HiddenMarkovModel().from_samples(distribution,n_components=self.K\
                ,X=self.preprocessed_data.copy())
                self.states=[np.array(hmm.predict(d.copy())) for d in self.preprocessed_data]
            else:
                print("Unrecognised or undefined clustering algorithm.")
                return -1
            self.experiment_progress=2

            
        #At the moment all this does is apply alpha filtering for HMM clustering
        def postprocess_data(self):
            self.postprocessed_data=self.preprocessed_data
            if self.alpha is not None:
                if (self.alg == "HMM"):
                    self.alpha_filter(data="post")
                    self.states=[s[q] for (s,q) in zip(self.states,self.quasistationary)]
            self.experiment_progress=3

        #return the state transition matrix
        #NEEDS FIXING FOR NEW LIST BASED FORMAT
        def transmat(self,**kwargs):
            from DorrRegime import transmat
            if self.states is not None:
                return transmat(self.states,**kwargs)
            else: print("No states found from which to calculate transition matrix")
        
        def get_centroids(self):
            if self.states is None:
                print("Need states to get centroids")
                return None
            else:
                D=self.postprocessed_data[0].data.shape[1]
                centroids=np.zeros([self.K,D])
                for k in range(self.K):
                    centroids[k]=np.concatenate(self.postprocessed_data)[np.concatenate(self.states)==k].mean(axis=0)
                return centroids
                
            
        #Intended for internal use by get_var_ratio and test_significance
        def _compute_var_ratio(self,data,states):
            if np.ndim(data)!=2:
                print("data must be rank 2")
                return None
            T,N=data.shape
            Ks=np.unique(states)
            intracluster_var=0
            centroid_means=np.zeros([len(Ks),N])
            
            for k in Ks:
                #compute the occupation weighted intracluster variance
                state_occupation=len(data[states==k])/T
                intracluster_var+=sum(data[states==k].var(axis=0))*state_occupation
                #compute centre of each cluster
                centroid_means[k]=data[states==k].mean(axis=0)
            
            #intercluster variance of cluster centroids
            intercluster_var=sum(centroid_means.var(axis=0))
            
            #var_ratio is variance between clusters divided by weighted variance
            #within clusters. For totally overlapping clusters this would be 0, for
            #perfect movement between discrete points it would be infinite
            var_ratio=intercluster_var/intracluster_var
            return var_ratio
        
        #Return the inter to intra cluster variance ratio of the clustering
        def get_var_ratio(self):
            if((self.postprocessed_data is None)|(self.states is None)):
                print("states and/or preprocessed_data not found")
            else:
                return self._compute_var_ratio(np.concatenate(self.postprocessed_data),np.concatenate(self.states))

        def get_significance(self,nsynth,bootstrap=100,repeat_rate=1,return_synth_ratios=False):
            from DorrSynthetic import autocorrelated_like
            import time
            
            var_ratio=self.get_var_ratio()
            
            MC_var_ratios=np.zeros(nsynth)
            
            for n in range(nsynth):
                #setup experiment to match original
                MCexp=ClusteringExperiment()
                MCexp.setup_like(self)
                MCexp.preprocess_data()
                
                np.random.seed(int(time.time())+n) #(necessary because hmm will reset seed to 0 on every loop!)
                #replace preprocessed data with synthetic copy
                MCexp.preprocessed_data=[autocorrelated_like(d,1)[...,0] for d in MCexp.preprocessed_data]
                
                #cluster and store the variance ratio
                MCexp.cluster()
                MCexp.postprocess_data()
                MC_var_ratios[n]=MCexp.get_var_ratio()
            #calculate the resulting p value of the clustering
            pval=sum(MC_var_ratios>var_ratio)/nsynth
            
            #estimate the error via bootstrapping of MC_var_ratios:
            pvals=np.zeros(bootstrap)
            for i in range(bootstrap):
                pvals[i]=sum((np.random.choice(MC_var_ratios,nsynth,replace=True))>var_ratio)/nsynth
            self.pval=pval
            self.pval_err=pvals.std()
            
            if return_synth_ratios:
                return (pval,pvals.std(),MC_var_ratios)
            else:
                return (pval,pvals.std())
            
        
        #This method plots the clusters resulting from fitting, taking a rank 3
        #cube of generating atmospheric fields as an input
        def plot_clusters(self,generating_data):
            import iris.plot as iplt
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            from DorrRegime import reg_lens
            
            #filter subsampled dates from generating data:
            if self.subsample_rate is not None:
                generating_data=generating_data[::self.subsample_rate]
                
            #filter quasistaionary days from generating data:
            if self.alpha is not None:
                generating_data=generating_data[self.quasistationary]
                
            #Create a cube of the K cluster mean composites
            cluster_cube=get_cluster_cube(generating_data,self.states)
            
            #Set up the figure
            plt.figure(figsize=(20,10))
            proj=ccrs.Orthographic(0, 90)
            
            #Plot the clusters:
            for i in range(self.K):
                ax=plt.subplot(3,self.K,i+1,projection=proj)
            
                clevs = np.linspace(-abs(cluster_cube.data).max(),abs(cluster_cube.data).max(), 21)

                ax.coastlines()
                ax.set_global()
                plot=iplt.contourf(cluster_cube[i], levels=clevs, cmap=plt.cm.RdBu_r)
                ax.set_title(f'{self.alg} cluster {i+1}', fontsize=16)
        
            #Plot intercluster std as a spatial field
            for i in range(self.K):
                ax=plt.subplot(3,self.K,self.K+i+1,projection=proj)
            
                std = generating_data[self.states==i].collapsed('time', iris.analysis.STD_DEV)
                clevs = np.linspace(0,std.data.max(), 21)

                ax.coastlines()
                ax.set_global()
                plot=iplt.contourf(std, levels=clevs, cmap=plt.cm.BuPu)
                ax.set_title(f'{self.alg} cluster {i+1}', fontsize=16)
                plt.colorbar(mappable=plot)
                
            #plot lifetime histograms
            L,S=reg_lens(self.states)
            for i in range(self.K):
                ax=plt.subplot(3,self.K,2*self.K+i+1)
                ax.hist(L[S==i],bins=np.linspace(0.5,50.5,51),label=f"mean lifetime = {(L[S==i]).mean():.1f} days")
                ax.legend()
            
            if self.alpha is not None:
                if self.subsample_rate is not None:
                    plt.suptitle(f"{self.alg} in {self.PCnum} PCs, with K={self.K}, sample rate {self.subsample_rate} days, alpha={self.alpha}",fontsize=24)
                else:
                    plt.suptitle(f"{self.alg} in {self.PCnum} PCs, with K={self.K}, alpha={self.alpha}",fontsize=24)
            elif self.subsample_rate is not None:
                    plt.suptitle(f"{self.alg} in {self.PCnum} PCs, with K={self.K}, sample rate {self.subsample_rate} days",fontsize=24)
            else:
                plt.suptitle(f"{self.alg} in {self.PCnum} PCs, with K={self.K},",fontsize=24)




"""
MC_Cluster_Experiment

Inputs: 
        NSynth - The number of synthetic datasets to construct and cluster.
        
        NAlgRep -The number of times to repeat the clustering algorithm for each
        dataset with different random seeds.
        
        PC_data - a list of iris cubes storing principal components. Each cube
        in the list will be treated as being temporally continuous, representing
        ,for example, an entire season
        
        PCnum - The number of principal components to use for clustering
        
        K - The number of clusters to which the data is fitted.
        
        alg - A string specifying the clustering algorithm to use; either
        'Kmeans' or 'HMM'.
        
        alpha - An optional parameter which keeps only the fraction 'alpha'
        of slowest evolving points for clustering. For HMM the filtering is 
        only applied at the post processing step due to the need for evenly
        spaced data.
        
        sbsp - An optional subsampling to apply to the data before clustering
        
        numpy_input - A flag to enable numpy input instead of iris cube input.
        Not Extensively Tested.
        
        shuffle - If shuffle is true, the list of input data to the clustering
        algorithm will be sampled randomly with replacement for each NAlgRep.
        The same will be done for the synthetic data. This allows the
        reproducibility of the clustering to be assessed.
        

Methods: run_real_experiment - Initialises and runs instances of the
         ClusteringExperiment class using the passed input parameters.
         
         make_synth_data - Creates NSynth sets of synthetic data to be used for
         clustering. These are derived from the raw data and preprocessed 
         in an identical manner to the true data.
        
         run_synth_experiments - Initialises and runs instances of the
         ClusteringExperiment class using the synthetic data, with the passed
         input parameters.
        
         get_real_var_ratios - Returns the variance ratio of every 
         ClusteringExperiment performed on the real data
        
         get_synth_var_ratios - Returns the variance ratio of every 
         ClusteringExperiment performed on the synthetic data
        
         get_real_repeatability - Returns a repeatability metric of the distance
         between cluster centroids for fits found on the same underlying
         real data.
        
         get_synth_repeatability -  Returns a repeatability metric of the distance
         between cluster centroids for fits found on the same underlying
         synthetic data.
        
         forget_data - Deletes large data arrays from the MC_Cluster_Experiment
         in order to free up memory
         
Attributes:
    
        data - The original input data (not a copy, a pointer)
                
        PCnum - The number of PCs being used
        
        K - The number of clusters
        
        alg - The chosen algorithm
        
        alpha - The value of alpha being used
        
        subsample_rate - the subsample rate being used
        
        NSynth - The number of synthetic datasets created
        
        NAlgRep - The number of times to repeat clusterings with different
        random seeds
        
        np - A flag for if numpy input data is being used
        
        shuffle - A flag that determines whether data cubes are resampled prior
        to repeated fitting. 
        If false, then the repeatability score measures the
        classifiability; how robust the exact same fitting is to random seeds,
        which assesses the condition of the fit. 
        If true, then the repeatability score measures the reproducibility;
        to what extent are the found clusters specific properties of that 
        particular dataset, as opposed to more global properties of
        the underlying system.
        
        RealExps - A list containing all the ClusteringExperiments based on 
        real data.
        
        synth - The synthetic data generated by make_synth_data()
        
        MCExps - A list of lists, containing all the ClusteringExperiments
        performed on each of the nsynth synthetic datasets.
        
        real_var_ratio - an array of variance ratios for each of the
        ClusteringExperiments in RealExps
        
        MC_var_ratios - a 2d array of variance ratios for each of the 
        ClusteringExperiments in MCExps.
  
"""         
                
class MC_Cluster_Experiment:
    
        def __init__(self,NSynth=None,NAlgRep=None,PC_data=None,PCnum=None,K=None,alg=None,alpha=None,sbsp=None,numpy_input=False,shuffle=False):
            self.data=PC_data
            self.PCnum=PCnum
            self.K=K
            self.alg=alg
            self.alpha=alpha
            self.subsample_rate=sbsp
            self.NSynth=NSynth
            self.NAlgRep=NAlgRep
            self.np=numpy_input
            self.shuffle=shuffle
            
            self.RealExps=[]
            self.synth=None
            self.MCExps=[]
            self.real_var_ratio=None
            self.MC_var_ratios=None
            
            if np.shape(self.data[0])[0]*np.shape(self.data[0])[1]*NAlgRep*NSynth>20.0*10**9:
                print("Warning: This configuration uses a dangerously large amount of memory")
                print("Using forgetful_synth=True is recommended")
                
        #repeats clustering on real data NAlgRep times, and appends experiment to list
        def run_real_experiment(self):
            import time
            for n in range(self.NAlgRep):
                
                #Either does or does not randomise the years based on shuffle
                if self.shuffle:
                    np.random.seed(int(time.time())+n)
                    data=np.random.choice(self.data,len(self.data)) #randomly resamples the years
                    
                else:
                    data=self.data  
                
                RealExp=ClusteringExperiment(PC_data=data,PCnum=self.PCnum,
                    K=self.K,alg=self.alg,alpha=self.alpha,sbsp=self.subsample_rate,numpy_input=self.np)
                
                RealExp.preprocess_data()
                RealExp.cluster()
                RealExp.postprocess_data()
                self.RealExps.append(RealExp)
                
        #creates synthetic data to be used for synthetic clustering
        def make_synth_data(self):
            from DorrSynthetic import autocorrelated_like
            if self.np:
                self.synth=autocorrelated_like(self.data[:,:self.PCnum],copies=self.NSynth)
                
            else:
                synths=[autocorrelated_like(np.array(d.data)[:,:self.PCnum],copies=self.NSynth) for d in self.data]
                self.synth=[[array[:,:,i] for array in synths] for i in range(self.NSynth)]         
        
        
        def run_synth_experiments(self,forgetful_synth=False):
            import time
            for member in self.synth:
                MC_repeats=[]
                for n in range(self.NAlgRep):
                    np.random.seed(int(time.time())+n) #(necessary because hmm will reset seed to 0 on every loop!)

                    #Either does or does not randomise the years based on 
                    #value of shuffle
                    if self.shuffle:
                         data=np.random.choice(member,len(member)) #randomly resamples the years
                    
                    else:
                         data=member
                    
                    MCExp=ClusteringExperiment(PC_data=data,PCnum=self.PCnum,
                    K=self.K,alg=self.alg,alpha=self.alpha,sbsp=self.subsample_rate,numpy_input=True)
                    
                    MCExp.preprocess_data()
                    MCExp.cluster()
                    MCExp.postprocess_data()
                                               
                    MC_repeats.append(MCExp)
                    
                self.MCExps.append(MC_repeats)
            
        #returns the var ratios calculated for the real data
        def get_real_var_ratios(self):
            self.real_var_ratio=np.array([Exp.get_var_ratio() for Exp in self.RealExps])
            return self.real_var_ratio
        
        def get_synth_var_ratios(self):
            var_ratios=np.zeros([self.NSynth,self.NAlgRep])
            for i,SingleMCExp in enumerate(self.MCExps):
                for j,RealisationExp in enumerate(SingleMCExp):
                    var_ratios[i,j]=RealisationExp.get_var_ratio()
            self.MC_var_ratios=var_ratios
            return var_ratios
           
        #Assesses the robustness of the clustering to random seeds, based on
        #the RMSE between separate clustering 
        #instances.
        def get_real_classifiability(self):
            
            c=0
            for Clustering1,Clustering2 in itertools.permutations(self.RealExps,2):
                
                centroids1=Clustering1.get_centroids()
                centroids2=Clustering2.get_centroids()
                    
                A=RMSE_matrix(centroids1,centroids2)
                #c is the highest RMSE value that gives the best correlation
                #between centroid1[i] and centroid2[j]
                c+=np.max(np.min(A,axis=1),axis=0)
            
            N=len(self.RealExps)
            c=c/(N*(N-1)) #turn sum into average
            self.real_classifiability=c
            return c
                    
        def get_synth_classifiability(self):
            self.synth_classifiability=[]
            for MCExp in self.MCExps:
                c=0
                for Clustering1,Clustering2 in itertools.permutations(MCExp,2):
                    
                    centroids1=Clustering1.get_centroids()
                    centroids2=Clustering2.get_centroids()
                        
                    A=RMSE_matrix(centroids1,centroids2)
                    #c is the highest RMS value that gives the best correlation
                    #between centroid1[i] and centroid2[j]
                    c+=np.max(np.min(A,axis=1),axis=0)
                N=len(MCExp)
                c=c/(N*(N-1)) #turn sum into average
                self.synth_classifiability.append(c)
            return self.synth_classifiability
    
        def forget_data(self):
            self.data=0
            self.preprocessed_data=0
            self.postprocessed_data=0
            self.synth=0
            for Exp in self.RealExps:
                Exp.data=0
                Exp.preprocessed_data=0
                Exp.postprocessed_data=0
                
            for MCExp in self.MCExps:
                for RealisationExp in MCExp:
                    RealisationExp.data=0
                    RealisationExp.preprocessed_data=0
                    RealisationExp.postprocessed_data=0
        
