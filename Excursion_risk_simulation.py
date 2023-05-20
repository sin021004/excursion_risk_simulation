import numpy as np
import matplotlib.pyplot as plt
import random
import sympy as sp
from sympy.utilities.lambdify import lambdify

class Brownian():
    
    def __init__(self, initial = 0):
        """
        Initialize Brownian Motion
        Input: Initial starting price
        Output: NA
        """
        assert (type(initial)==float or type(initial)==int or initial is None)
        #Basic Setup
        self.initial = initial
        self.walk = []

        #Delta Excursions
        self.delta_interval = []
        self.last_delta = []
        self.delta_diff = []
        
        #Truncated Variation
        self.TV_intervals = []

    def gen_random_walk(self, step_num):
        """
        Generates list of random walk results
        Input: Number of steps
        Output: List of random walks
        """
        walk = np.ones(step_num)*self.initial
        for i in range(1, step_num):
            next = np.random.choice([1, -1])
            walk[i] = walk[i - 1] + next
        self.walk = walk
        return walk
    
    def gen_normal_walk(self, step_num):
        """
        Generates list of using normal distribution walk results
        Input: Number of steps
        Output: List of random walks following normal distribution
        """
        walk = np.ones(step_num)*self.initial
        for i in range(1, step_num):
            next = np.random.normal()
            walk[i] = walk[i - 1] + next
        self.walk = walk
        return walk

    def calculate_delta_excursions(self, delta_val):
        """
        Calculate lists of delta excursions including lists of delta interval, last delta, and delta difference
        Input: Delta excursions value
        Output: N/A
        """
        begin = 0
        passed = False
        last_delta = 0
        for i in range(len(self.walk)):
            if passed and self.walk[i] <= 0:
                cur_interval = [begin, i]
                self.delta_interval.append(cur_interval)
                self.last_delta.append(last_delta)
                self.delta_diff.append(i - last_delta)
                begin = i
                passed = False
            if self.walk[i] >= delta_val:
                passed = True
                last_delta = i


    def print_delta_interval(self):
        """
        Print a list of delta intervals
        Input: N/A
        Output: list of delta intervals
        """
        print("Delta Interval: ", self.delta_interval)
    
    def print_last_delta(self):
        """
        Print a list of last deltas
        Input: N/A
        Output: list of last deltas
        """
        print("Last Delta: ", self.last_delta)
    
    def print_delta_diff(self):
        """
        Print a list of delta difference
        Input: N/A
        Output: list of delta difference
        """
        print("Delta Diff: ", self.delta_diff)
    
    def calculate_truncated_variation(self, constant_C):
        """
        Calculate truncated variation intervals
        Input: Constant C
        Output: list of truncated variation intervals
        """
        if constant_C <= 0:
            print("Warning, C has to be greater than 0")
            exit(1)
        
        highest = float('-inf')
        lowest = float('inf')
        for i in range(len(self.walk)):      
            if len(self.TV_intervals) == 0 or len(self.TV_intervals) % 2 == 0:
                #Looking for positive TV
                if self.walk[i] <= lowest:
                    lowest = self.walk[i]
                if self.walk[i] >= lowest + constant_C:
                    self.TV_intervals.append(i)
                    lowest = float('inf')
            else:
                #Looking for negative TV
                if self.walk[i] >= highest:
                    highest = self.walk[i]
                if self.walk[i] <= highest - constant_C:
                    self.TV_intervals.append(i)  
                    highest = float('-inf')
        
    
    def print_truncated_variation(self):
        """
        Print a list of truncated variation intervals
        Input: N/A
        Output: list of truncated variation intervals
        """
        print("Truncated Variation Interval: ", self.TV_intervals)


class Fractional_Brownian():

    def __init__(self, initial=0, Hurst=0.5):
        """
        Initialize Fractional Brownian Motion
        Input: Initial starting price and Hurst paramets                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        Output: NA
        """
        assert (type(initial)==float or type(initial)==int or initial is None)
        assert (type(Hurst)==float)
        if Hurst > 1:
            print("Hurst parameter has to be between 0 and 1")
            exit(1)
        self.initial = initial
        self.Hurst = Hurst
        self.walk = []

        self.delta_interval = []
        self.last_delta = []
        self.delta_diff = []

        self.TV_intervals = []

        self.delta_excursion_limit = 0
    
    def gen_walk(self, step_num):
        """
        Generates list of walk results based on Hurst parameters=
        Input: Number of steps
        Output: List of walks
        """
        walk = np.ones(step_num)*self.initial
        next = np.random.choice([1, -1])
        walk[1] = walk[0] + next
        for i in range(2, step_num):
            if walk[i - 1] > walk[i - 2] :
                next = np.random.choice([1, -1], 1, replace=False, p=[self.Hurst, (1 - self.Hurst)])
            else: 
                next = np.random.choice([-1, 1], 1, replace=False, p=[self.Hurst, (1 - self.Hurst)])
            walk[i]  = walk[i - 1] + next[0]
        self.walk = walk
        return walk   
    
    
    def calculate_delta_excursions(self, delta_val):
        """
        Calculate lists of delta excursions including lists of delta interval, last delta, and delta difference
        Input: Delta excursions value
        Output: N/A
        """
        begin = 0
        passed = False
        last_delta = 0
        for i in range(len(self.walk)):
            if passed and self.walk[i] <= 0:
                cur_interval = [begin, i]
                self.delta_interval.append(cur_interval)
                self.last_delta.append(last_delta)
                self.delta_diff.append(i - last_delta)
                begin = i
                passed = False
            if self.walk[i] >= delta_val:
                passed = True
                last_delta = i

    def print_delta_interval(self):
        """
        Print a list of delta intervals
        Input: N/A
        Output: list of delta intervals
        """
        print("Delta Interval: ", self.delta_interval)
    
    def print_last_delta(self):
        """
        Print a list of last deltas
        Input: N/A
        Output: list of last deltas
        """
        print("Last Delta: ", self.last_delta)
    
    def print_delta_diff(self):
        """
        Print a list of delta difference
        Input: N/A
        Output: list of delta difference
        """
        print("Delta Diff: ", self.delta_diff)
    
    def calculate_truncated_variation(self, constant_C):
        """
        Calculate truncated variation intervals
        Input: Constant C
        Output: list of truncated variation intervals
        """
        if constant_C <= 0:
            print("Warning, C has to be greater than 0")
            exit(1)
        highest = float('-inf')
        lowest = float('inf')
        for i in range(len(self.walk)):      
            if len(self.TV_intervals) == 0 or len(self.TV_intervals) % 2 == 0:
                #Looking for positive TV
                if self.walk[i] <= lowest:
                    lowest = self.walk[i]
                if self.walk[i] >= lowest + constant_C:
                    self.TV_intervals.append(i)
                    lowest = float('inf')
            else:
                #Looking for negative TV
                if self.walk[i] >= highest:
                    highest = self.walk[i]
                if self.walk[i] <= highest - constant_C:
                    self.TV_intervals.append(i)  
                    highest = float('-inf')
    
    def print_truncated_variation(self):
        """
        Print a list of truncated variation intervals
        Input: N/A
        Output: list of truncated variation intervals
        """
        print("Truncated Variation Interval: ", self.TV_intervals)

    def calculate_P_value(self):
        """
        Estimate P value by taking a limit on delta excursions length where delta value goes to 0
        Input: N/A
        Output: Range of P Value
        """
        d = sp.Symbol('d')
        def delta_excursions_length(self, delta):
            begin = 0
            passed = False
            delta_interval = []
            d = sp.Symbol('d')
            n = sp.Symbol('n')
            f = lambdify([n, d], self.walk[int(n)] - d, modules='numpy')
            for i in range(len(self.walk)):
                if passed and self.walk[i] <= 0:
                    cur_interval = [begin, i]
                    delta_interval.append(cur_interval)
                    begin = i
                    passed = False
                if float(self.walk[i]) > delta:
                #if f(i, delta) >= 0:
                    passed = True
            return len(delta_interval)

        L = sp.limit(delta_excursions_length(self, d), d, 0)
        return L.evalf().round(3)


    

    # def calculate_delta_excursion_limit(self):
    #     """
    #     Calculate the limit of delta excursion length as delta_val goes to 0
    #     Input: None
    #     Output: The limit of delta excursion length
    #     """
    #     delta_excursion_lengths = []
    #     for delta_val in np.logspace(-10, 0, num=100, base=10):
    #         self.calculate_delta_excursions(delta_val)
    #         delta_excursion_lengths.append(sum(self.delta_diff))
    #         self.delta_interval = []
    #         self.last_delta = []
    #         self.delta_diff = []
    #     self.delta_excursion_limit = np.mean(delta_excursion_lengths)
    #     return self.delta_excursion_limit
    
    # def print_delta_excursion_limit(self):
    #     print("Delta Excursion Limit: ", self.delta_excursion_limit) 

        


        
        

    


def main():
    # b = Brownian()
    # for i in range(5):
    #     plt.plot(b.gen_random_walk(10000))
    # #plt.show()

    # b.calculate_delta_excursions(1)
    # b.print_delta_interval()
    # b.print_last_delta()
    # b.print_delta_diff()

    # b.calculate_truncated_variation(1)
    # b.print_truncated_variation()

    # plt.show()


    fb = Fractional_Brownian(0, 0.50)
    for i in range(1):
        plt.plot(fb.gen_walk(50000))

    # fb.calculate_delta_excursions(-1)
    # fb.print_delta_interval()
    # fb.print_last_delta()
    # fb.print_delta_diff()

    # fb.calculate_truncated_variation(3)
    # fb.print_truncated_variation()

    fb.calculate_P_value()




    plt.show()


main()
    