import json


def f1skew(acc):
    return (2*acc)/(1 + acc)

a = [0.1*i for i in range(11)]
b = [f1skew(0.1*i) for i in range(11)]

f = [{'acc': a[i], 'f1':b[i] }for i in range(11)]

print(json.dumps(f))