clc, clear all
q01s=0.3;
q02s=0.5;
k11=1.15;
k22=1.3;
F1=0.5;
F2=0.8;

h1s=(q01s/k11)^2
h2s=((q01s+q02s)/k22)^2

k1=k11/((2*sqrt(h1s)))
k2=k22/(2*(sqrt(h2s)))

h10=h1s
h20=h2s

input=[0 q01s; 20 1.05*q01s; 40 0.95*q01s; 60 1.02*q01s; 80 1.02*q01s]
