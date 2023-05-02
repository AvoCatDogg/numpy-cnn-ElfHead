#여기서는 코드에 사용될 activation function들을 정의해두었다.

import numpy as np

#Relu 함수를 정의
class Relu:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    #이 클래스에 가중치가 있는지에 대한 코드. Fasle
    def has_weights(self):
        return self.has_units

    #forward_propagate과정에서의 Relu를 정의. savecash=False임으로 Z값을 저장하지 않음.
    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        return np.where(Z >= 0, Z, 0)

    #back_propagete과정에서의 Relu정의. 여기서는 cache dict에 'Z' 에 해당하는 값을 Z에 불러옴
    def back_propagate(self, dA):
        Z = self.cache['Z']
        return dA * np.where(Z >= 0, 1, 0)

#Softmax 함수를 정의.
class Softmax:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    #이 클래스에 가중치가 있는지에 대한 코드. Fasle
    def has_weights(self):
        return self.has_units

    #forward_propagate과정에서의 Softmax를 정의. savecash=False임으로 Z값을 저장하지 않음.
    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        return e / np.sum(e, axis=0, keepdims=True)

    #back_propagete과정에서의 Relu정의. 여기서는 cache dict에 'Z' 에 해당하는 값을 Z에 불러옴
    def back_propagate(self, dA):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))

#Elu 함수를 정의 Elu는 self.params로 alpha라는 parameter를 가진다.
class Elu:
    def __init__(self, alpha=1.2):
        self.cache = {}
        self.params = {
            'alpha': alpha
        }
        self.has_units = False

    #이 클래스에 가중치가 있는지에 대한 코드. Fasle
    def has_weights(self):
        return self.has_units

    #forward_propagate과정에서의 Elu를 정의. savecash=False임으로 Z값을 저장하지 않음.
    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        return np.where(Z >= 0, Z, self.params['alpha'] * (np.exp(Z) - 1))

    #back_propagete과정에서의 Elu정의. 여기서는 cache dict에 'Z' 에 해당하는 값을 Z에 불러오며, parameter alpha를 불러옴. 
    def back_propagate(self, dA):
        alpha = self.params['alpha']
        Z = self.cache['Z']
        return dA * np.where(Z >= 0, 1, self.forward_propagate(Z, alpha) + alpha)

#Selu함수를 정의. alpha와 selu_lambda parameter을 가진다.
class Selu:
    def __init__(self, alpha=1.6733, selu_lambda=1.0507):
        self.params = {
            'alpha' : alpha,
            'lambda' : selu_lambda
        }
        self.cache = {}
        self.has_units = False

    #이 클래스에 가중치가 있는지에 대한 코드. Fasle
    def has_weights(self):
        return self.has_units

    #forward_propagate과정에서의 Selu를 정의. savecash=False임으로 Z값을 저장하지 않음.
    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        return self.params['lambda'] * np.where(Z >= 0, Z, self.params['alpha'] * (np.exp(Z) - 1))

    #back_propagete과정에서의 Selu정의. 여기서는 cache dict에 'Z' 에 해당하는 값을 Z에 불러오며, parameter alpha를 불러옴. 
    def back_propagate(self, dA):
        Z = self.cache['Z']
        selu_lambda, alpha = self.params['lambda'], self.params['alpha']
        return dA * selu_lambda*np.where(Z >= 0, 1, alpha*np.exp(Z))