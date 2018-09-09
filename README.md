# Recurrent-Neural-Network
Vanilla Recurrent Neural Network to synthesize text sentences after training of optional text such as books.


## Learning of Lord of the Rings
I trained the network on the Lord of the Rings book "The Fellowship of the Ring":

![](https://github.com/johndah/Recurrent-Neural-Network/blob/master/Learning%20Curve.png)

Sample of synthezised text:

Sequence iteration: 1591581, Epoch: 39, Smooth loss: 42.37

	"Frown
	of the San concougly of the wish; it was his masted lon Rif meding? Farser wo king any but any'I wiscore of and lear treen
	down awar but neat..'

	Galfot his from time"

I uploaded a set of parameters in directory that I trained for 40 epochs (1 772 039 sequence iterations).

## Learning of Python code
I also wanted to test how well the network could learn to write Python code, for this I created a text file including all Python code from my Github repositories to train on (as much it may seem, the network would ideally be feeded with way more data to generalize better), with a sequence length seqLength = 100. Here is the result:

![](https://github.com/johndah/Recurrent-Neural-Network/blob/master/Learning%20Curve%20Python.png)

Sample of synthezised text:

Epoch: 1999, Lowest smooth loss: 69.298

	print('\n'	
	validm.anarate((self.neur', p[i, :, getattr(self, weight, sequen(sigmaNoise))

	hPrev = self.computeAcs:	

	# Epoch multiLayerPerceppighnolNuch intseNt arn.f)
	tol in range(6):
		print('RBMIn self.plotProwTr' + etcalerataplet itimm.Tration%(self.W[layer-2].T).T
	self.lmbda):
	
	self.Xtrais.append('grax}']
	for i in range(len(self.weights))]
		self.X0, fol aur(-10tind)) - X

	def multeile(reck.reshot inpued(dJdW     X itines': 1 = c1) Circliss))
		self.Ntrain = array(diall)
		self.X0e-rasedTept(self.iteration = 1 + '\nW ' + 1

## Tips for use
Begin training from scrach by entering "He" or "Random" as attribute weightInit, or enter "Load" while ensuring weights are saved (set attribute saveParameters to "True") to begin from trained state.

Fewer hidden neurons will require larger weight init variance (automatically done with He initialization).

Be careful with activating rmsProp as gradients may expload, good values for <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma,&space;\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma,&space;\eta" title="\gamma, \eta" /></a>
 are 0.9, 0.001.
