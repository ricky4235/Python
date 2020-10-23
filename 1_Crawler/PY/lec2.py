# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:42:45 2020

@author: 11004076
"""

nums = [-2,1,-3,4,-1,2,1,-5,4]


sum = 0
for i in nums:
    if sum + i > sum:
        sum = sum + i
    else:
        sum = sum
print(sum)

