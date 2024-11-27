
import pandas as pd
import numpy as np
from scipy.stats import norm

class CLOModel():
    def __init__(self, **kwargs):
        self.setup(**kwargs) # set up parameters, by running function setup() on initialization
        self.load_default() # load default rates
        
    def setup(self, **kwargs):
        self.V0 = 100 # asset value period 0
        self.rf = 0.0384 # risk free interest rate
        self.beta = 0.8
        self.em = 0.07
        self.c = self.beta*self.em # risk premium parameter     
        self.sigma_m = self.beta*0.14 # market level variance (0.8 stems from beta)
        self.sigma_j = 0.25 # firm level variance
        self.sigma = (self.sigma_m ** 2 + self.sigma_j ** 2) ** 0.5 # total variance parameter
        self.T = 5 # time to maturity
        self.m = 1000 # number of steps (granularity of process)
        self.n = 200000 # number of CLO simulations for rating agencies
        self.q = 1000 # number of CLO simulations for interim market value updating
        self.j = 125 # number of loans
        self.rating = 'B' # rating for each loan
        self.default = self.load_default() # default probability table
        self.noncallperiod = 2
        
        
        for key,val in kwargs.items():
            setattr(self,key,val) 
            
    def load_default(self):
        '''Loads S&P Cumulative Default Rates for 1981-2023 as a dataframe'''
        df = pd.read_excel('data\S&P Global Corporate Average Cumulative Default Rates.xlsx', sheet_name='adjust', header=2, index_col='Rating')
        df = df.loc[:'B', df.columns[:8]]
        return df
    
    def load_historical_refinance_data(self):
        '''Loads historical refinance data as a dataframe'''
        df = pd.read_excel('data\Historical refinancing data.xlsx', sheet_name='Data', header=0)
        return df
    
    def load_historical_spreads(self):
        '''Loads historical spreads as a dataframe'''
        df = pd.read_excel('data\Historical Spreads.xlsx', sheet_name='DataEU', header=0)
        return df


    def GBM_fig(self):
        '''Geometric Brownian Motion for illustration purpose
        RETURNS:
        Numpy array of dimensions (1 + number of simulations, total steps)
        '''
        # see: https://quantpy.com.au/stochastic-calculus/simulating-geometric-brownian-motion-gbm-in-python/

        # (1) define calculation parts
        dt = 1 / self.m # calculate each step (total time / frequency)
        drift = ((self.c + self.rf) - 0.5 * self.sigma ** 2)*dt
        
        # (2) draw and prepare array
        np.random.seed(1705) # set seed
        #W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(self.n, self.m)) # draw normal dist
        W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(self.T*self.m+1, self.n, self.j+1)) # draw normal dist

        diff = self.sigma_m * W[:,:,0].reshape((self.T*self.m+1, self.n, 1)) + self.sigma_j * W[:,:,1:]
        incr = drift + diff
        incr[0,:,:] = 0
        
        # (3) use cumulative product (over rows) to calculate simulation paths, and multiply by initial value V0
        return self.V0 * np.exp(incr.cumsum(axis=0))

    def GBM(self, risk_neutral=False):
        '''Geometric Brownian Motion for CLO model        
        RETURNS:
        Numpy array of dimensions (time to maturity + 1, # of simulations, # of loans)
        '''
        
        # (1) define calculation parts
        if risk_neutral: drift = self.rf - 0.5 * self.sigma ** 2
        else: drift = (self.c + self.rf) - 0.5 * self.sigma ** 2

        # (2) draw and prepare array
        np.random.seed(2020) # set seed
        W = np.random.normal(loc=0, scale=1, size=(self.T+1, self.n, self.j+1)) # draw normal distribution

        # (3) calculate increments and take sum
        diff = self.sigma_m * W[:,:,0].reshape((self.T+1, self.n, 1)) + self.sigma_j * W[:,:,1:]
        incr = drift + diff
        incr[0,:,:] = 0 # period t=1 has no drift/diffusion yet
        
        return self.V0 * np.exp(incr.cumsum(axis=0))
    

    def GBM_shock(self, risk_neutral=False,shock=None):

        
        # (1) define calculation parts
        if risk_neutral: drift = self.rf - 0.5 * self.sigma ** 2
        else: drift = (self.c + self.rf) - 0.5 * self.sigma ** 2

        # (2) draw and prepare array
        np.random.seed(2020) # set seed
        W = np.random.normal(loc=0, scale=1, size=(self.T+1, self.n, self.j+1)) # draw normal distribution
        marketshocks=W[:,:,0]
        marketshocks[1, :] = shock

        # (3) calculate increments and take sum
        diff = self.sigma_m * marketshocks.reshape((self.T+1, self.n, 1)) + self.sigma_j * W[:,:,1:]
        incr = drift + diff
        incr[0,:,:] = 0 # period t=1 has no drift/diffusion yet
        
        return self.V0 * np.exp(incr.cumsum(axis=0))






    
    def ExpSimulation(self, remainding_years=None, Start_values=None):

        years = remainding_years
        dt = years
        drift = (self.rf - 0.5 * self.sigma ** 2) * dt
        firm_values = np.array(Start_values).reshape(1, -1)  # Reshape for broadcasting
        np.random.seed(1705)  # Set seed

        # Draw from normal distribution with shape (q, j+1) to match dimensions
        W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(self.q, self.j + 1))

        # Calculate the diff based on W
        diff = self.sigma_m * W[:, 0].reshape(self.q, 1) + self.sigma_j * W[:, 1:]
        incr = drift + diff

        # Broadcasting firm_values with exp(incr) for correct shape
        result = firm_values * np.exp(incr)
        B = self.face_value()

        return np.minimum(result,B)
        




    def NonCall_GBM(self, risk_neutral=True):
        '''Geometric Brownian Motion for CLO model        
        RETURNS:
        Numpy array of dimensions (self.noncallperiod + 1, # of simulations, # of loans)
        '''
        
        # (1) define calculation parts
        if risk_neutral: drift = self.rf - 0.5 * self.sigma ** 2
        else: drift = (self.c + self.rf) - 0.5 * self.sigma ** 2

        # (2) draw and prepare array
        np.random.seed(2020) # set seed
        W = np.random.normal(loc=0, scale=1, size=(self.noncallperiod+1, self.n, self.j+1)) # draw normal distribution

        # (3) calculate increments and take sum
        diff = self.sigma_m * W[:,:,0].reshape((self.noncallperiod+1, self.n, 1)) + self.sigma_j * W[:,:,1:]
        incr = drift + diff
        incr[0,:,:] = 0 # period t=1 has no drift/diffusion yet
        result = self.V0 * np.exp(incr.cumsum(axis=0))

        return result[-1] #Return the firm values right before the noncall period ends
    
    def ExpectationSimulation(self, risk_neutral=False, NonCallValues=None):
        '''Geometric Brownian Motion for CLO model        
        RETURNS:
        Numpy array of dimensions (num_simulations, self.T-self.noncallperiod+1, # of simulations, # of loans)
        '''
    
        if risk_neutral: 
            drift = self.rf - 0.5 * self.sigma ** 2
        else: 
            drift = (self.c + self.rf) - 0.5 * self.sigma ** 2

        results = np.zeros((self.q, self.T-self.noncallperiod+1, self.j))
    
        for i in range(self.q):  
            W = np.random.normal(loc=0, scale=1, size=(self.T-self.noncallperiod+1, self.j+1)) # draw normal distribution
            diff = self.sigma_m * W[:,0].reshape((self.T-self.noncallperiod+1, 1)) + self.sigma_j * W[:,1:]
            incr = drift + diff
            incr[0,:] = 0 # period t=1 has no drift/diffusion yet

            result = np.zeros((self.T-self.noncallperiod+1, self.j))
            result[0, :] = NonCallValues
            result[1:, :] = result[0, :] * np.exp(incr[1:, :].cumsum(axis=0))
            results[i] = result
    
        return results

    def Geometric_Brownian_motion(self, risk_neutral=True, Start_values=None,number_of_years=None):
        
        if risk_neutral: 
            drift = self.rf - 0.5 * self.sigma ** 2
        else: 
            drift = (self.c + self.rf) - 0.5 * self.sigma ** 2

        
        np.random.seed(2024)
        results = np.zeros((self.q, number_of_years+1, self.j))
        W = np.random.normal(loc=0, scale=1, size=(number_of_years+1, self.j+1,self.q)) # draw normal distribution
        diff = self.sigma_m * W[:,0].reshape((number_of_years+1, 1,self.q)) + self.sigma_j * W[:,1:,:]
        incr = drift + diff
        incr[0,:,:] = 0 # period t=1 has no drift/diffusion yet
        result = np.zeros((number_of_years+1, self.j,self.q))
        result[0, :, :] = Start_values
        result[1:, :,:] = result[0, :, :] * np.exp(incr[1:, :, :].cumsum(axis=0))

        return result

    def SPV_value(self, risk_neutral=False):
        '''SPV terminal values
        '''
        # (1) draw asset paths
        if risk_neutral: V = self.GBM(risk_neutral=True)[-1,:,:]
        else: V = self.GBM()[-1,:,:] # take only terminal values
        
        # (2) calculate minimum and take sum
        B = self.face_value()
        CF = np.minimum(V, B)
        CF_sum = np.sum(CF, axis=1) # take sum over firms j = 1, ..., J
        
        return np.sort(CF_sum)
    


    def SPV_value_shock(self, risk_neutral=False,shock=None):
        '''SPV terminal values
        '''
        # (1) draw asset paths
        if risk_neutral: V = self.GBM_shock(risk_neutral=True, shock=shock)[-1,:,:]
        else: V = self.GBM_shock(shock=shock)[-1,:,:] # take only terminal values
        
        # (2) calculate minimum and take sum
        B = self.face_value()
        CF = np.minimum(V, B)
        CF_sum = np.sum(CF, axis=1) # take sum over firms j = 1, ..., J
        
        return np.sort(CF_sum)
    
    def face_value(self, rating='B'):
        '''Face value from cumulative default probability table
        RETURNS:
        Face value B
        '''
        
        def_prob = self.default.loc[rating, self.T] / 100 # cumulative default probability from table        
        
        return self.V0 / np.exp( - norm.ppf(def_prob) * self.sigma * np.sqrt(self.T)
                                 - ( (self.c + self.rf) - 0.5 * self.sigma ** 2 ) * self.T )
    
    def market_value_underlying(self, B):

        
        d_1 = (np.log(self.V0 / B) + (self.rf + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d_2 = d_1 - self.sigma*np.sqrt(self.T)

        market_val = B*np.exp(-self.rf*self.T)*norm.cdf(d_2)+self.V0*norm.cdf(-d_1)


        return B*np.exp(-self.rf*self.T)*norm.cdf(d_2)+self.V0*norm.cdf(-d_1)
    
    
    def result_table_with_equity(self):
        '''Creates result table
        RETURNS:
        DataFrame with model simulation results
        '''
        # (1) create DataFrame
        df = self.default
        df = df.loc[:'B', [self.T]] # select only maturity T
        df.rename(columns={self.T:'default probability'}, inplace=True)
        equity_row = pd.DataFrame({'default probability': [np.nan]}, index=['Equity'])
        df = pd.concat([df, equity_row])

        SPV_values = self.SPV_value()
        df['aggregate face value'] = np.quantile(SPV_values, np.where(df['default probability'].isna(), 1, df['default probability'].values / 100))
        SPV_values_Q = self.SPV_value(risk_neutral=True) # risk neutral SPV values

        Debt = df.loc[df.index[df.index != 'Equity'][-1], 'aggregate face value']

        for k in df.index:  # for each rating
            Bk = df.loc[k, 'aggregate face value']  # Use the precomputed aggregate face value
            if k == 'Equity':
                df.loc[k, 'aggregate market value'] = np.maximum(0, SPV_values_Q-Debt).mean() * np.exp(-self.rf * self.T)+df.loc[df.index[df.index != 'Equity'][-1], 'aggregate market value']
            else:
                df.loc[k, 'aggregate market value'] = np.minimum(SPV_values_Q, Bk).mean() * np.exp(-self.rf * self.T)

        df['face value'] = df['aggregate face value'] - df['aggregate face value'].shift(1)
        df.loc['AAA', 'face value'] = df.loc['AAA', 'aggregate face value']

        df['%'] = (df['face value'] / df['face value'].sum()) * 100

        df['market value'] = df['aggregate market value'] - df['aggregate market value'].shift(1)
        df.loc['AAA', 'market value'] = df.loc['AAA', 'aggregate market value']

        df['Price'] = df['market value'] / df['face value'] * 100

        # (6) yield and spread columns
        df['Yield'] = 1 / self.T * np.log(df['face value'] / df['market value']) * 100
        df['Spread'] = df['Yield'] - self.rf * 100

        return df



    def result_table_with_equity_shock(self,shock=None):
        '''Creates result table
        RETURNS:
        DataFrame with model simulation results
        '''
        # (1) create DataFrame
        df = self.default
        df = df.loc[:'B', [self.T]] # select only maturity T
        df.rename(columns={self.T:'default probability'}, inplace=True)
        equity_row = pd.DataFrame({'default probability': [np.nan]}, index=['Equity'])
        df = pd.concat([df, equity_row])

        SPV_values = self.SPV_value()
        df['aggregate face value'] = np.quantile(SPV_values, np.where(df['default probability'].isna(), 1, df['default probability'].values / 100))
        SPV_values_Q = self.SPV_value_shock(risk_neutral=True, shock=shock) # risk neutral SPV values

        Debt = df.loc[df.index[df.index != 'Equity'][-1], 'aggregate face value']

        for k in df.index:  # for each rating
            Bk = df.loc[k, 'aggregate face value']  # Use the precomputed aggregate face value
            if k == 'Equity':
                df.loc[k, 'aggregate market value'] = np.maximum(0, SPV_values_Q-Debt).mean() * np.exp(-self.rf * (self.T-1))+df.loc[df.index[df.index != 'Equity'][-1], 'aggregate market value']
            else:
                df.loc[k, 'aggregate market value'] = np.minimum(SPV_values_Q, Bk).mean() * np.exp(-self.rf * (self.T-1))

        df['face value'] = df['aggregate face value'] - df['aggregate face value'].shift(1)
        df.loc['AAA', 'face value'] = df.loc['AAA', 'aggregate face value']

        df['%'] = (df['face value'] / df['face value'].sum()) * 100

        df['market value'] = df['aggregate market value'] - df['aggregate market value'].shift(1)
        df.loc['AAA', 'market value'] = df.loc['AAA', 'aggregate market value']

        df['Price'] = df['market value'] / df['face value'] * 100

        # (6) yield and spread columns
        df['Yield'] = 1 / (self.T-1) * np.log(df['face value'] / df['market value']) * 100

        return df
    