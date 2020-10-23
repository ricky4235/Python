# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:12:17 2020

@author: 11004076
"""

"""
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        tmp = nums[0]
        max_ = tmp
        n = len(nums)
        for i in range(1,n):

                max_ = max(max_, tmp, tmp+nums[i], nums[i])
                tmp = nums[i]
        return max_
nums = [2,7,11,15]
target = 9     
        
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        dict = {}
        for index, num in enumerate(nums):
            if target - num not in dict:
                dict[num] = index
            else:
                return [dict[target - num], index]

If  is odd, print Weird
If  is even and in the inclusive range of  to , print Not Weird
If  is even and in the inclusive range of  to , print Weird
If  is even and greater than , print Not Weird


#!/bin/python3

import math
import os
import random
import re
import sys

def 
if __name__ == '__main__':
    n = int(input().strip())
    if n%2 == 0:
        if n in range(2, 6):
            print('N')
        elif n in range(2, 21):
            print('We')
        elif n>20:
            print('Not')
    else:
        print('Wed')
    
    
    
    
    
 result = result.reshape(3,3)


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)
    """
# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import randn
from numpy.linalg import inv, qr



x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])

dot_ = x.dot(y)  # 等價於np.dot(x, y)
print(dot_)  # dot是正常的矩陣相乘的方法  求內積

print(np.ones(3))  #  生成一個元素全為1的陣列 [ 1.  1.  1.]

# 要指定完整的shape（完整的行數和列數）的矩陣陣列
o4 = np.ones( (2, 3), dtype = int)  
print(o4) 
'''
[[1 1 1]
 [1 1 1]]
 
 zeros() 
'''

# seed() 
print( np.random.seed(12345) )  # None

X = randn(5, 5)
mat = X.T.dot(X)   # X.T x的轉置 與 x 的 dot內積
print(mat)
print(np.linalg.inv(mat))
mat.dot(inv(mat))
q, r = qr(mat)

def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c
numpysum(10)

print(np.eye(3))

arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)  # astype 型別轉換
float_arr.dtype

print(float_arr.dtype)  # float64
print(float_arr)  # [ 1.  2.  3.  4.  5.]
#
a = data[1:2] #第二行  array([[81, 12, 27, 39, 83, 15]])
b = data[2:3] #第三行  array([[49, 39, 16, 54, 93, 14]])
index=np.isin(a,b) #返回布林值 array([[False, False, False,  True, False, False]])
array=a[~index]  #~:按位取反運算符：對數據的每個二進制位取反,即把1變為0,把0變為1 。
array
