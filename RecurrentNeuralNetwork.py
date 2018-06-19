'''
@author: John Henry Dahlberg

2018-06-19
'''

import sklearn.preprocessing
from numpy import *
from copy import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

class RecurrentNeuralNetwork(object):

    def __init__(self, attributes=None):
        if not attributes:
            raise Exception('Dictionary argument "attributes" is required.')

        self.__dict__ = attributes

        bookData, charToInd, indToChar = self.loadCharacters()
        self.bookData = bookData
        self.charToInd = charToInd
        self.indToChar = indToChar
        self.K = len(indToChar)

        self.x0 = '.'
        self.h0 = zeros((self.nHiddenNeurons, 1))

        self.weights = ['W', 'V', 'U', 'b', 'c']
        self.gradients = ['dLdW', 'dLdV', 'dLdU', 'dLdB', 'dLdC']
        self.numGradients = ['gradWnum', 'gradVnum', 'gradUnum', 'gradBnum', 'gradCnum']

        self.sizes = [(self.nHiddenNeurons, self.nHiddenNeurons), (self.K, self.nHiddenNeurons), \
                      (self.nHiddenNeurons, self.K), (self.nHiddenNeurons, 1), (self.K, 1)]

        # Weight initialization
        for weight, gradIndex in zip(self.weights, range(len(self.gradients))):
            if self.sizes[gradIndex][1] > 1:
                if self.weightInit == 'He':
                    self.initSigma = sqrt(2 / sum(self.sizes[gradIndex]))
                else:
                    self.initSigma = 0.01
                setattr(self, weight, self.initSigma*random.randn(self.sizes[gradIndex][0], self.sizes[gradIndex][1]))
            else:
                setattr(self, weight, zeros(self.sizes[gradIndex]))

        '''
        for weight in self.weights[:3]:
            setattr(self, weight, array(loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)))
        for weight in self.weights[3:]:
            setattr(self, weight, array([loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)
        '''


    def loadCharacters(self):
        with open(self.textFile, 'r') as f:
            lines = f.readlines()
        bookData = ''.join(lines)
        characters = []
        [characters.append(char) for sentences in lines for char in sentences if char not in characters]
        k = len(characters)
        indicators = array(range(k))

        indOneHot = self.toOneHot(indicators)

        charToInd = dict((characters[i], array(indOneHot[i])) for i in range(k))
        indToChar = dict((indicators[i], characters[i]) for i in range(k))

        return bookData, charToInd, indToChar

    def adaGrad(self):

        xChars = self.bookData[:self.seqLength]
        yChars = self.bookData[1:self.seqLength + 1]
        x = self.seqToOneHot(xChars)
        y = self.seqToOneHot(yChars)
        hPrev = deepcopy(self.h0)

        p, h, a = self.forwardPass(x, hPrev)
        self.backProp(x, y, p, h, a)

        smoothLoss = self.computeCost(p, y)
        smoothLosses = []
        lowestSmoothLoss = copy(smoothLoss)


        if self.plotProcess:
            fig = plt.figure()
            constants = 'Max Epochs: ' + str(self.nEpochs) \
                        + '\n# Hidden neurons: ' + str(self.nHiddenNeurons) \
                        + '\nWeight initialization: ' + str(self.weightInit) \
                        + '\n' + r'$\sigma$ = ' + "{:.2e}".format(self.initSigma) \
                        + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) \
                        + '\n' + 'Sequence length: ' + str(self.seqLength) \
                        + '\n' + '# training characters in text:' + '\n' + str(len(self.bookData)) \
                        + '\n' + 'AdaGrad: ' + str(self.adaGradSGD) \
                        + '\n' + 'RMS Prop: ' + str(self.rmsProp)

            if self.rmsProp:
                constants += '\n' + r'$\gamma$ = ' + "{:.2e}".format(self.gamma) \



        m = []
        for grad in self.gradients:
            m.append(getattr(self, grad)**2)

        seqIteration = 0
        seqIterations = []

        for epoch in range(0, self.nEpochs):

            hPrev = deepcopy(self.h0)

            for e in range(0, len(self.bookData)-self.seqLength-1, self.seqLength):
                xChars = self.bookData[e:e+self.seqLength]
                yChars = self.bookData[e+1:e+self.seqLength + 1]
                x = self.seqToOneHot(xChars)
                y = self.seqToOneHot(yChars)

                p, h, a = self.forwardPass(x, hPrev)
                hPrev = deepcopy(h[-1])

                loss = self.computeCost(p, y)
                smoothLoss = .999 * smoothLoss + 0.001 * loss

                self.backProp(x, y, p, h, a)

                epsilon = 1e-10

                if self.rmsProp:
                    cM = self.gamma
                    cG = 1 - self.gamma
                else:
                    cM, cG, = 1, 1

                for grad, weight, gradIndex in zip(self.gradients, self.weights, range(len(self.gradients))):
                    if self.adaGradSGD:
                        m[gradIndex] = cM * m[gradIndex] + cG*getattr(self, grad)**2
                        sqrtInvM = (m[gradIndex]+epsilon)**-0.5
                        updatedWeight = getattr(self, weight) - self.eta * multiply(sqrtInvM, getattr(self, grad))
                    else:
                        updatedWeight = getattr(self, weight) - self.eta * getattr(self, grad)
                    setattr(self, weight, updatedWeight)

                if e % (self.seqLength*1e2) == 0:
                    seqIterations.append(seqIteration)
                    smoothLosses.append(smoothLoss)
                    x0 = self.bookData[e]
                    sequence = self.synthesizeText(x0, hPrev, self.lengthSynthesizedText)
                    print('\nSequence iteration: ' + str(seqIteration) + ', Epoch: ' + str(epoch) + ', Epoch process: ' \
                          + str('{0:.2f}'.format(e/len(self.bookData)*100)) + '%' + ', Smooth loss: ' + str('{0:.2f}'.format(smoothLoss)))
                    print('    ' + sequence)

                    if smoothLoss < lowestSmoothLoss:
                        lowestSmoothLoss = copy(smoothLoss)
                        '''
                        
                        for weight in self.weights:
                            savetxt(str(weight) + '.txt', getattr(self, weight), delimiter=',')

                        savetxt('seqIterations.txt', seqIterations, delimiter=',')
                        savetxt('smoothLosses.txt', smoothLosses, delimiter=',')
                        '''

                    if self.plotProcess:
                        plt.clf()
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(top=0.85)
                        anchored_text = AnchoredText(constants, loc=1)
                        ax.add_artist(anchored_text)

                        plt.title('Text prediction loss of Recurrent Neural Network')
                        plt.ylabel('Smooth loss')
                        plt.xlabel('Sequence iteration')
                        plt.plot(seqIterations, smoothLosses, LineWidth=2)
                        plt.grid()
                        plt.pause(0.1)

                seqIteration += 1

            bestWeights = []

            for weight in self.weights[:3]:
                bestWeights.append(array(loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)))
            for weight in self.weights[3:]:
                bestWeights.append(array([loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)

            weightsTuples = [(self.weights[i], bestWeights[i]) for i in range(len(self.weights))]

            weights = dict(weightsTuples)
            bestSequence = self.synthesizeText(x0, hPrev, self.lengthSynthesizedTextBest, weights)

            print('\n\nEpoch: ' + str(epoch) + ', Lowest smooth loss: ' + str(lowestSmoothLoss))
            print('    ' + bestSequence)

    def forwardPass(self, x, hPrev, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)

        tau = len(x)

        h = [hPrev]
        a = []
        p = []
        for t in range(0, tau):
            a.append(dot(weights['W'], h[t]) + dot(weights['U'], x[t]) + weights['b'])
            h.append(self.tanh(a[t]))
            o = dot(weights['V'], h[t+1]) + weights['c']
            p.append(self.softmax(o))

        return p, h, a

    def synthesizeText(self, x0, hPrev, seqLength, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)

        xNew = []
        xNext = x0

        for t in range(0, seqLength):
            x = [array([self.charToInd.get(xNext)]).T]
            p, h, a = self.forwardPass(x, hPrev, weights)
            hPrev = deepcopy(h[-1])

            cp = cumsum(p[-1])
            rand = random.uniform()
            sample = abs(cp - rand).argmin()
            xNext = self.indToChar.get(sample)
            xNew.append(xNext)

        sequence = ''.join(xNew)

        return sequence

    def backProp(self, x, y, p, h, a):
        tau = len(x)

        # Initialize gradients
        for grad, weight in zip(self.gradients, self.weights):
            setattr(self, grad, zeros(getattr(self, weight).shape))

        dLdO = []

        for t in range(tau):
            dPdO = diag(p[t][:, 0]) - dot(p[t], p[t].T)
            dLdO.append(-(1/(dot(y[t].T, p[t])))*dot(y[t].T, dPdO))
            self.dLdV += dot(dLdO[t].T, h[t+1].T)
            self.dLdC += dLdO[t].T

        dLdH = [zeros((1, self.nHiddenNeurons))]*tau
        dLdA = [zeros((1, self.nHiddenNeurons))]*tau
        dLdH[-1] = dot(dLdO[-1], self.V)
        dLdA[-1] = dot(dLdH[-1], diag(1 - self.tanh(a[t-1][:, 0])**2))

        for t in range(tau - 2, -1, -1):
            dLdH[t] = dot(dLdO[t], self.V) + dot(dLdA[t+1], self.W)
            dLdA[t] = dot(dLdH[t], diag(1 - self.tanh(a[t][:, 0])**2))

        for t in range(tau):
            self.dLdW += dot(dLdA[t].T, h[t].T)
            self.dLdU += dot(dLdA[t].T, x[t].T)
            self.dLdB += dLdA[t].T

        # Clip gradients
        for grad in self.gradients:
            setattr(self, grad, maximum(minimum(getattr(self, grad), 5), -5))


    def computeCost(self, p, y):
        # Cross entropy loss

        tau = len(y)
        loss = 0
        for t in range(tau):
            loss -= sum(log(dot(y[t].T, p[t])))

        return loss

    def toOneHot(self, x):
        binarizer = sklearn.preprocessing.LabelBinarizer()
        binarizer.fit(range(max(x.astype(int)) + 1))
        X = array(binarizer.transform(x.astype(int))).T

        return X

    def seqToOneHot(self, x):
        X = [array([self.charToInd.get(xt)]).T for xt in x]

        return X

    def seqToOneHotMatrix(self, x):
        xInd = self.seqToOneHot(x)
        X = concatenate(xInd, axis=1)

        return X


    def tanh(self, x):
        return (exp(x) - exp(-x))/(exp(x) + exp(-x))

    def softmax(self, s):
        p = zeros(s.shape)

        for i in range(s.shape[0]):
            p[i, :] = exp(s[i, :])

        p /= sum(p, axis=0)

        return p

    def computeGradsNumSlow(self, x, y, hPrev):
        hStep = 1e-4

        for numGrad, weight in zip(self.numGradients, self.weights):
            setattr(self, numGrad, zeros(getattr(self, weight).shape))

        for numGrad, grad, gradIndex in zip(self.numGradients, self.gradients, range(len(self.numGradients))):
            if getattr(self, grad).shape[1] == 1:
                for i in range(getattr(self, grad).shape[0]):
                    gradTry = deepcopy(getattr(self, grad))
                    gradTry[i, 0] -= hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], gradTry)
                    weights = dict(weightsTuples)
                    p, h, a = self.forwardPass(x, hPrev, weights)
                    c1 = self.computeCost(p, y)

                    gradTry = deepcopy(getattr(self, grad))
                    gradTry[i, 0] += hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], gradTry)
                    weights = dict(weightsTuples)
                    p, h, a = self.forwardPass(x, hPrev, weights)
                    c2 = self.computeCost(p, y)

                    updatedNumGrad = deepcopy(getattr(self, numGrad))
                    updatedNumGrad[i, 0] = (c2 - c1) / (2 * hStep)
                    setattr(self, numGrad, updatedNumGrad)
            else:
                iS = [0, 1, 2, getattr(self, grad).shape[0] - 3, getattr(self, grad).shape[0] - 2, getattr(self, grad).shape[0] - 1]
                jS = [0, 1, 2, getattr(self, grad).shape[1] - 3, getattr(self, grad).shape[1] - 2, getattr(self, grad).shape[1] - 1]
                for i in iS:  # range(self.W[layer].shape[0]):
                    for j in jS:  # range(self.W[layer].shape[1]):
                        gradTry = deepcopy(getattr(self, grad))
                        gradTry[i, j] -= hStep
                        weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], gradTry)
                        weights = dict(weightsTuples)
                        p, h, a = self.forwardPass(x, hPrev, weights)
                        c1 = self.computeCost(p, y)

                        gradTry = deepcopy(getattr(self, grad))
                        gradTry[i, j] += hStep
                        weightsTuples = [(self.weights[i], copy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], gradTry)
                        weights = dict(weightsTuples)
                        p, h, a = self.forwardPass(x, hPrev, weights)
                        c2 = self.computeCost(p, y)

                        updatedNumGrad = deepcopy(getattr(self, numGrad))
                        updatedNumGrad[i, j] = (c2 - c1) / (2 * hStep)
                        setattr(self, numGrad, updatedNumGrad)

    def testComputedGradients(self):

        xChars = self.bookData[:self.seqLength]
        yChars = self.bookData[1:self.seqLength+1]
        x = self.seqToOneHot(xChars)
        y = self.seqToOneHot(yChars)

        epsilon = 1e-20
        hPrev = self.h0
        p, h, a = self.forwardPass(x, hPrev)
        self.backProp(x, y, p, h, a)

        differenceGradients = []
        differenceGradientsSmall = []
        self.computeGradsNumSlow(x, y, hPrev)

        for numGrad, grad, gradIndex in zip(self.numGradients, self.gradients, range(len(self.numGradients))):
            gradObj = deepcopy(getattr(self, grad))
            numGradObj = deepcopy(getattr(self, numGrad))

            differenceGradients.append(abs(gradObj - numGradObj) / maximum(epsilon, (abs(gradObj) + abs(numGradObj))))

            if gradObj.shape[1] > 1:
                # Only calculate first and last three rows and columns
                differenceGradientsSmall.append(zeros((6, 6)))

                iS = [0, 1, 2, gradObj.shape[0] - 3, gradObj.shape[0] - 2, gradObj.shape[0] - 1]
                jS = [0, 1, 2, gradObj.shape[1] - 3, gradObj.shape[1] - 2, gradObj.shape[1] - 1]

                for i in range(6):
                    for j in range(6):
                        differenceGradientsSmall[gradIndex][i, j] = "{:.2e}".format(differenceGradients[gradIndex][iS[i], jS[j]])
            else:
                differenceGradientsSmall.append(zeros((1, 6)))

                bS = [0, 1, 2, gradObj.shape[0] - 3, gradObj.shape[0] - 2, gradObj.shape[0] - 1]

                for i in range(6):
                    differenceGradientsSmall[gradIndex][0, i] = "{:.2e}".format(differenceGradients[gradIndex][bS[i]][0])

            print('\nAbsolute differences gradient ' + grad + ':')
            print(differenceGradientsSmall[gradIndex])
            # print(pMatrix(differenceWSmall[layer]))


def pMatrix(array):

    rows = str(array).replace('[', '').replace(']', '').splitlines()
    rowString = [r'\begin{pmatrix}']
    # rowString += [r'  \num{' + r'} & \num{'.join(row.split()) + r'}\\' for row in rows]
    for row in rows:
        rowString += [r'  \num{' + r'} & \num{'.join(row.split()) + r'}\\']

    rowString += [r'\end{pmatrix}']

    return '\n'.join(rowString)


def main():

    attributes = {
        'textFile': 'LordOfTheRings.txt',
        'adaGradSGD': True,
        'weightInit': 'He',
        'eta': .1,
        'nHiddenNeurons': 100,
        'seqLength': 25,
        'lengthSynthesizedText': 200,
        'lengthSynthesizedTextBest': 1000,
        'rmsProp': False,
        'gamma': 0.9,
        'nEpochs': 30,
        'plotProcess': True
    }

    rnn = RecurrentNeuralNetwork(attributes)
    rnn.testComputedGradients()
    rnn.adaGrad()

if __name__ == '__main__':
    random.seed(1)
    main()
