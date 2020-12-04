#Hiperparâmetros à priori
mu0 = 10
alpha0 = 2
beta0 = 25*alpha0
lambda0 = beta0/(4*(alpha0-1))
print(f"mu0 = {mu0}\nlambda0 = {lambda0}\nalpha0 = {alpha0}\nbeta0 = {beta0}\n\n")

#Estatísticas Suficientes
xn = 8.307849
sn = 7.930452
n = 10

#Hiperparâmetros à posteriori
mu1 = (lambda0*mu0+n*xn)/(lambda0+n)
lambda1 = lambda0+n
alpha1 = alpha0+n/2
beta1 = beta0 + sn/2 + (n*lambda0*(xn-mu0)**2)/(2*(lambda0+n))
print(f"mu1 = {mu1}\nlambda1={lambda1}\nalpha1 = {alpha1}\nbeta1 = {beta1}\n\n")

#Intervalo de credibilidade
gamma = 0.95
from scipy.stats import t
c = t.ppf((1+gamma)/2,df=2*alpha1)
c2 = (lambda1*alpha1/beta1)**(-1/2)
a = -c*c2+mu1
b = c*c2+mu1

print(f"P({a} < mu < {b}) = {gamma}")



