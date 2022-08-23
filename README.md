# Neural Network

## Simulating the costs ##

The aim of this project is studying the evolution of the epidemics in a population of $1000$ individuals using the $SIR$ model and finding the configuration that leads to the minimum cost.

Firstly, we simulated the number of infections in the first $60$ days, starting from $S_0 = 999$, $I_0 = 1$. Every individual in $S$ has a small probability $p$ to get infected by an individual in $I$ and all the individuals in $I$ at time $t$ get removed at time $t + 1$. So, we could say that the number of infected people at time $t + 1$ is a random variable distributed as a binomial distribution

$$ I(t+1)\sim Bin(S_t, 1-(1-p)^{I(t)}). $$

Consequently, the number of susceptible people at time $t+1$ is

$$ S(t+1)=S(t)-I(t+1). $$

Secondly, containment measures that vary the value of $p$ between $0.003$ and $0.0005$ were introduced. The evolution of the epidemics was simulated as described above for $11$ different values of $p$ in the interval of interest. Since the sanitary cost is unitary for every infected person the average cost of health $\mathrm{E}_p[c_h]= \mathrm{E}_p[I(60)+R(60)]= N-\mathrm{E}_p[S(60))]$.

Lastly, since for every containment measure level it is possible to write the economic cost $c_e(p)=\left(\frac{0.003}{p}\right)^{9}-1$, we computed corresponding average cost as the sum of cost of health and economic cost.

## Training the Neural Network ##

We used a non-linear neural network with:
 - one input layer with $2$ nodes, one for the argument of the function and a bias node;
 - one hidden layer consisting of $H$ neurons plus a bias node;
 - one output node that will take the value of the approximating function.

![](images/NN.png?raw=true)

*Here is an example of a non-linear neural network with $H=3$ hidden nodes.*

To manage nonlinearity, we had to use an activation function to transform the input of the second layer to its output. We have chosen the sigmoid function

$$ \Sigma(x)=\frac{1}{1+e^{-x}}, $$

since it is a typical choice and we obtained good results using it.

We noticed that the values of $p$ were very small and those of $c$ were large. These may lead to the explosion of the weights ($w$ and $k$) in the back-propagation algorithm, therefore we decided to normalize both the vector $p$ and the vector $c$ before running the algorithm.

### Tuning Parameters ###

Firstly, we had to choose an updating rule for the stepsize considering that approximately $3$ million iterations were needed. We decided to use a rule of the type

$$ \mu(m)=\frac{A}{B+m}. $$

We did a grid search on the possible values of $A$ and $B$ and we obtained good results using $A=5000$ and $B=100000$. Smaller values of $\mu$ leads to a non-convergent algorithm and larger values often had as a result an explosion of the weights. 

![](images/mu.png?raw=true)

*Value of the stepsize $\mu$ with increasing $m$.*

Secondly, we observed how varies the value of $SSE$, where

$$ SSE = \sum_{p=1}^{N_p}(y_p-o_p)^2, $$

and $o_p$ is the output of the network corresponding to point $p$. We noticed that in the first $10000$ iterations the evolution is not regular (the figure below clearly states it), therefore we decided the run at least $10000$ iterations of the algorithm.

![](images/SSE.png?raw=true)

*As shown on the left, the values of $SSE$ in the first $10000$ iterations are unstable, after this, as shown on the right, the evolution becomes more regular.*

Lastly, we had to choose a stopping criterion and we decided to evaluate the relative differences, that is

$$ \frac{SSE_{old}-SSE_{new}}{SSE_{old}}\leq tol. $$

We grid searched the optimal value of $tol$. In particular, we observed that for values smaller than $10^{-7}$ the algorithm took too much time to reach the convergence. Moreover, we observed a worst approximation by using values bigger than $10^{-7}$ as shown in the figure below.

![](images/11_3.png?raw=true)

*In the figures are shown the approximating functions obtained by using $tol = 10^{-6}$ on the left and $tol = 10^{-7}$ on the right.*

## Conclusions ##

After tuning all the parameters, the minimum cost achieved was about $-173000$ which is out of the domain since it is negative. To solve the problem, we tried to increment the hyperparameter $H$ (the number of hidden nodes) without achieving any reasonable results (as shown in the following figure).

![](images/11_H.png?raw=true)

*In the figures is shown the minimum cost achieved by using a neural network with $H=5$ hidden nodes on the left and $H=9$ nodes on the right.*

Finally, we observed that the first $2$ points had a much bigger value than the others and, probably, by fitting those $2$ points too well the model ends up not being accurate when doing predictions. Accordingly, we tried running the back-propagation algorithm with the $9$ nodes in the interval $[0.001;0.003]$, and we get the final results in the figure below. 

![](images/finale.png?raw=true)

*Final model trained using 9 points instead of 11.*

By evaluating the resulting function, we found that the minimum cost is approximately $612$ and correspond to a value of $p=0.00166$.

