'''
@author: John Henry Dahlberg

2018-06-24
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

        self.seqIterations = loadtxt('seqIterations.txt', delimiter=",", unpack=False)
        self.smoothLosses = loadtxt('smoothLosses.txt', delimiter=",", unpack=False)

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

        self.lossMomentum = 1e-3

        '''
        Uncomment to load saved weights
        
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
        print('Unique characters:\n' + str(characters))
        k = len(characters)
        indicators = array(range(k))

        indOneHot = self.toOneHot(indicators)

        charToInd = dict((characters[i], array(indOneHot[i])) for i in range(k))
        indToChar = dict((indicators[i], characters[i]) for i in range(k))

        return bookData, charToInd, indToChar

    def adaGrad(self):
        smoothLoss = None
        smoothLosses = []

        if self.plotProcess:
            fig = plt.figure()
            constants = 'Max Epochs: ' + str(self.nEpochs) + ' (' + str(round(1772039/39 * self.nEpochs)) + ' seq. iter.)' \
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
        for weight in self.weights:
            m.append(zeros(getattr(self, weight).shape))

        seqIteration = 0
        seqIterations = []

        for epoch in range(0, self.nEpochs):

            hPrev = deepcopy(self.h0)

            for e in range(0, len(self.bookData)-self.seqLength-1, self.seqLength):

                xChars = self.bookData[e:e+self.seqLength]
                yChars = self.bookData[e+1:e+self.seqLength + 1]
                x = self.seqToOneHot(xChars)
                y = self.seqToOneHot(yChars)

                p, h = self.forwardProp(x, hPrev)
                self.backProp(x, y, p, h)

                loss = self.computeLoss(p, y)
                if not smoothLoss:
                    smoothLoss = loss
                    lowestSmoothLoss = copy(smoothLoss)
                smoothLoss = (1 - self.lossMomentum) * smoothLoss + self.lossMomentum * loss

                if e % (self.seqLength*3e2) == 0:
                    seqIterations.append(seqIteration)
                    smoothLosses.append(smoothLoss)

                    x0 = self.bookData[e]
                    sequence = self.synthesizeText(x0, hPrev, self.lengthSynthesizedText)

                    print('\nSequence iteration: ' + str(seqIteration) + ', Epoch: ' + str(epoch) + ', Epoch process: ' \
                          + str('{0:.2f}'.format(e/len(self.bookData)*100)) + '%' + ', Smooth loss: ' + str('{0:.2f}'.format(smoothLoss)))
                    print('    ' + sequence)

                    if smoothLoss < lowestSmoothLoss:
                        lowestSmoothLoss = copy(smoothLoss)
                        hPrevBest = hPrev
                        x0Best = x0
                        if self.saveParameters:
                            try:
                                for weight in self.weights:
                                    savetxt(weight + '.txt', getattr(self, weight), delimiter=',')

                                savetxt('seqIterations.txt', seqIterations, delimiter=',')
                                savetxt('smoothLosses.txt', smoothLosses, delimiter=',')
                            except Exception as e: 
                                print(e)

                    if self.plotProcess:
                        plt.clf()
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(top=0.85)
                        anchored_text = AnchoredText(constants, loc=1)
                        ax.add_artist(anchored_text)

                        plt.title('Text synthesization learning curve of Recurrent Neural Network')
                        plt.ylabel('Smooth loss')
                        plt.xlabel('Sequence iteration')
                        plt.plot(seqIterations, smoothLosses, LineWidth=2)
                        plt.grid()
                        plt.pause(0.1)

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
                        updatedWeight = deepcopy(getattr(self, weight)) - self.eta * deepcopy(getattr(self, grad))

                    setattr(self, weight, updatedWeight)

                hPrev = deepcopy(h[-1])

                seqIteration += 1

            bestWeights = []

            for weight in self.weights[:3]:
                bestWeights.append(array(loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)))
            for weight in self.weights[3:]:
                bestWeights.append(array([loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)

            weightsTuples = [(self.weights[i], bestWeights[i]) for i in range(len(self.weights))]

            weights = dict(weightsTuples)
            bestSequence = self.synthesizeText(x0Best, hPrevBest, self.lengthSynthesizedTextBest, weights)
            print('\n\nEpoch: ' + str(epoch) + ', Lowest smooth loss: ' + str(lowestSmoothLoss))
            print('    ' + bestSequence)

    def forwardProp(self, x, hPrev, weights={}):
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

        return p, h

    def synthesizeText(self, x0, hPrev, seqLength, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)
        y = []
        xChar = copy(x0)

        for t in range(0, seqLength):
            x = self.seqToOneHot(xChar)
            p, h = self.forwardProp(x, hPrev, weights)
            hPrev = deepcopy(h[-1])

            cp = cumsum(p[-1])
            rand = random.uniform()
            diff = cp - rand
            sample = [i for i in range(len(diff)) if diff[i] > 0][0]

            xChar = self.indToChar.get(sample)
            y.append(xChar)

        sequence = ''.join(y)

        return sequence

    def backProp(self, x, y, p, h):
        tau = len(x)

        # Initialize gradients
        for grad, weight in zip(self.gradients, self.weights):
            setattr(self, grad, zeros(getattr(self, weight).shape))

        dLdO = []

        for t in range(tau):
            dLdO.append(p[t].T - y[t].T)
            self.dLdV += dot(dLdO[t].T, h[t+1].T)
            self.dLdC += dLdO[t].T

        dLdAt = zeros((1, self.nHiddenNeurons))

        for t in range(tau - 1, -1, -1):
            dLdHt = dot(dLdO[t], self.V) + dot(dLdAt, self.W)
            dLdAt = dot(dLdHt, diag(1 - h[t+1][:, 0]**2))

            self.dLdW += dot(dLdAt.T, h[t].T)
            self.dLdU += dot(dLdAt.T, x[t].T)
            self.dLdB += dLdAt.T

        # Clip gradients
        if self.clipGradients:
            for grad in self.gradients:
                setattr(self, grad, maximum(minimum(getattr(self, grad), 5), -5))

    def computeLoss(self, p, y):
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
        exP = exp(s)
        p = exP/exP.sum()

        return p

    def computeGradsNumSlow(self, x, y, hPrev):
        hStep = 1e-4

        for numGrad, weight in zip(self.numGradients, self.weights):
            setattr(self, numGrad, zeros(getattr(self, weight).shape))

        for numGrad, weight, gradIndex in zip(self.numGradients, self.weights, range(len(self.numGradients))):
            if getattr(self, weight).shape[1] == 1:
                for i in range(getattr(self, weight).shape[0]):
                    weightTry = deepcopy(getattr(self, weight))
                    weightTry[i, 0] -= hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                    weights = dict(weightsTuples)
                    p, h = self.forwardProp(x, hPrev, weights)
                    c1 = self.computeLoss(p, y)

                    weightTry = deepcopy(getattr(self, weight))
                    weightTry[i, 0] += hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                    weights = dict(weightsTuples)
                    p, h = self.forwardProp(x, hPrev, weights)
                    c2 = self.computeLoss(p, y)

                    updatedNumGrad = deepcopy(getattr(self, numGrad))
                    updatedNumGrad[i, 0] = (c2 - c1) / (2 * hStep)
                    setattr(self, numGrad, updatedNumGrad)
            else:
                iS = [0, 1, 2, getattr(self, weight).shape[0] - 3, getattr(self, weight).shape[0] - 2, getattr(self, weight).shape[0] - 1]
                jS = [0, 1, 2, getattr(self, weight).shape[1] - 3, getattr(self, weight).shape[1] - 2, getattr(self, weight).shape[1] - 1]
                for i in iS:
                    for j in jS:
                        weightTry = deepcopy(getattr(self, weight))
                        weightTry[i, j] -= hStep
                        weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                        weights = dict(weightsTuples)
                        p, h = self.forwardProp(x, hPrev, weights)
                        c1 = self.computeLoss(p, y)

                        weightTry = deepcopy(getattr(self, weight))
                        weightTry[i, j] += hStep
                        weightsTuples = [(self.weights[i], copy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                        weights = dict(weightsTuples)
                        p, h = self.forwardProp(x, hPrev, weights)
                        c2 = self.computeLoss(p, y)

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
        p, h = self.forwardProp(x, hPrev)
        self.backProp(x, y, p, h)

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
            #print(pMatrix(differenceGradientsSmall[gradIndex]))


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
        'clipGradients': True,
        'weightInit': 'He',
        'eta': .1,
        'nHiddenNeurons': 100,
        'seqLength': 25,
        'lengthSynthesizedText': 200,
        'lengthSynthesizedTextBest': 1000,
        'rmsProp': False,
        'gamma': 0.9,
        'nEpochs': 40,
        'plotProcess': True,
        'saveParameters': False
    }

    if not attributes['adaGradSGD']:
        attributes['eta'] = 0.01*attributes['eta']

    rnn = RecurrentNeuralNetwork(attributes)
    #rnn.testComputedGradients()
    rnn.adaGrad()


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
