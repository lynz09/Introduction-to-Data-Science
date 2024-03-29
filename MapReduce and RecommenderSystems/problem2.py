from mrjob.job import MRJob
import logging
#-------------------------------------------------------------------------
'''
    Problem 2: 
    In this problem, you will use mapreduce framework to implement matrix multiplication. 
    
    Matrix Dataset:
    Suppose we have a dataset of two matrics A and B (matrix.csv), each line represents an element in matrix A or matrix B.
    For example, to represent a (2X3) matrix A = 1,2,3
                                                 4,5,6
        and a (3X2) matrix B = 1,-1
                               1,-1
                               1,-1
        We want to compute the matrix C = A X B (matrix multiplication)

        C =  6, -6
            15, -15 

        We will have the following input file:

        Matrix, Row_index, Column_index, Value, num_rows, num_columns
        -------------------------------------------------------------
          A   ,    1     ,      1      ,   1  ,   2     ,     2
          A   ,    1     ,      2      ,   2  ,   2     ,     2
          A   ,    1     ,      3      ,   3  ,   2     ,     2
          A   ,    2     ,      1      ,   4  ,   2     ,     2
          A   ,    2     ,      2      ,   5  ,   2     ,     2
          A   ,    2     ,      3      ,   6  ,   2     ,     2
          B   ,    1     ,      1      ,   1  ,   2     ,     2
          B   ,    1     ,      2      ,  -1  ,   2     ,     2
          B   ,    2     ,      1      ,   1  ,   2     ,     2
          B   ,    2     ,      2      ,  -1  ,   2     ,     2
          B   ,    2     ,      1      ,   1  ,   2     ,     2
          B   ,    2     ,      2      ,  -1  ,   2     ,     2
    
    Here num_rows (num_column) represents the number of rows (columns) in matrix C.

'''

#--------------------------
class MatMul(MRJob):
#--------------------------
    ''' 
        Given a matrix A and a matrix B, compute the product A*B = C (matrix multiplication)
    '''

    #----------------------
    def mapper(self, in_key, in_value):
        ''' 
            mapper function, which process a key-value pair in the data and generate intermediate key-value pair(s)
            Input:
                    in_key: the key of a data record (in this example, can be ignored)
                    in_value: the value of a data record, (in this example, it is a line of text string in the data file, check 'matrix.csv' for example)
            Yield: 
                    (out_key, out_value) :intermediate key-value pair(s). You need to design the format and meaning of the key-value pairs. These intermediate key-value pairs will be feed to reducers, after grouping all the values with a same key into a value list.
        '''
        
        #########################################
        ## INSERT YOUR CODE HERE

        # process input value
        #in_value = in_value.decode()

        line = in_value.strip()

        par = line.split(",")
        r, k = int(par[4]), int(par[5])

        #row, column of matrix C
        tag, row, columns, value = par[0], int(par[1]), int(par[2]), int(par[3])

        # generate output key-value pairs 
        if tag == 'A':
            for i in range(1,r+1):
                key = ("C",row,i)
                # values = (tag, row, columns, value)
                yield (key, (tag, columns, value))
        else:
            for j in range(1,k+1):
                key = ("C",j, columns)
                yield (key,(tag, row, value))

        #########################################


    #----------------------
    def reducer(self, in_key, in_values):
        ''' 
            reducer function, which processes a key and value list and produces output key-value pair(s)
            Input:
                    in_key: an intermediate key from the mapper
                    in_values: a list (generator) of values , which contains all the intermediate values with the same key (in_key) generated by all mappers
            Yield: 
                    (out_key, out_value) : output key-value pair(s). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # a = dict()
        # b = dict()
        # sum = 0
        # logging.error("a", in_values)
        # for line in in_values:
        #     logging.error("line1:%s",line)
        #     key, v0, v1, v2 = line[0], int(line[1]), int(line[2]), int(line[3])
        #     logging.warning("key1")
        #     if key == 'A':
        #         a[v1] = v2
        #     elif key == "B":
        #         logging.warning("b",key)
        #         b[v0] = v2
        # logging.warning("a b %s",a)
        # for g in a:
        #     for k, arr in a.items():
        #         mul = arr[0] * b[k][0]
        #         logging.warning("a[k] b[k]")
        #         sum += float(mul)
        #     yield (in_key, sum)
        #logging.warning("in_Key: %s %d %b",in_key[0],in_key[1], in_key[2])
        left_matrix = [(item[1], item[2]) for item in in_values if item[0] == 'A']
        right_matrix = [(item[1], item[2]) for item in in_values if item[0] == 'B']
        result = 0

        for item_L in left_matrix:
            for item_R in right_matrix:
                if item_L[0] == item_R[0]:
                    result += item_L[1] * item_R[1]

        if result != 0:
            yield (in_key, result)

            # Mlist = {}
        # Nlist = {}
        # for line in in_values:
        #     line = line.strip()
        #     key, i, val = line.split(',')
        #     i = int(i)
        #     val = float(val)
        #
        #     if key == 'A':
        #         if key in Mlist.keys():
        #             Mlist[in_key].append((i, val))
        #         else:
        #             Mlist[in_key] = [(i, val)]
        #     else:
        #         if key in Nlist.keys():
        #             Nlist[in_key].append((i, val))
        #         else:
        #             Nlist[in_key] = [(i, val)]
        #
        # for key, arr in Mlist.items():
        #     s = 0
        #     for j, val in enumerate(arr):
        #         s += val[1] * Nlist[in_key][j][1]
        #
        #     yield ("%s %s" % (key, s))





















        #########################################




#--------------------------------------------

''' TEST Problem 2: 
        Now you can test the correctness of all the above functions by typing `nosetests -v test2.py' in the terminal.  

        If your code passed all the tests, you will see the following message in the terminal:
            ----------- Problem 2 (15 points in total)-------------- ... ok
            (3 points) MatMul1x1 ... ok
            (3 points) MatMul1x2 ... ok
            (3 points) MatMul2x1 ... ok
            (3 points) MatMul2x2 ... ok
            (3 points) MatMul random ... ok
            ----------------------------------------------------------------------
            Ran 5 tests in 0.103s            
            OK

'''

#--------------------------------------------





