#!/usr/bin/env python

'''
    Task: Provide Simple Pricing Functions
    FINS3648
    License: ""
'''

'''Payments = r(PV)/(1-(1+yield)^-T)'''

notional = PV = 4000000000     #in cash currency
cash_yield = r = 0.03      #% per annum
timeToMaturity = n = 1     #in years

def loan_payment(notional, cash_yield, timeToMaturity):
    P = cash_yield*(notional)/(1-(1+cash_yield)**(-timeToMaturity))
    return P

print(loan_payment(notional, cash_yield, timeToMaturity)*timeToMaturity)



def zero_coupon_bond(notional, cash_yield, timeToMaturity):
    return notional / (1 + cash_yield) ** timeToMaturity

print(zero_coupon_bond(1000000, 0.009, 5))
print(zero_coupon_bond(notional, cash_yield, timeToMaturity))

print(loan_payment(notional, cash_yield, timeToMaturity))
print(loan_payment(notional, cash_yield, timeToMaturity)*timeToMaturity)
