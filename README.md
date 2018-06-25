# Recurrent-Neural-Network
Vanilla Recurrent Neural Network to synthesize text sentences after training of optional text such as books.

![](https://github.com/johndah/Recurrent-Neural-Network/blob/master/Learning%20Curve.png)

Begin training from scrach by entering "He" or "Random" as attribute weightInit, or enter "Load" while ensuring weights are saved (set attribute saveParameters to "True") to begin from trained state.

I uploaded a set of parameters in directory that I trained for 40 epochs (1 772 039 sequence iterations).

Fewer hidden neurons will require larger weight init variance (automatically done with He initialization).

Be careful with activating rmsProp as gradients may expload, good values for <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma,&space;\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma,&space;\eta" title="\gamma, \eta" /></a>
 are 0.9, 0.001.
