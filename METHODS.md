# Model Formulation
We formulate measles transmission as a partially observed continuous time Markov chain. The latent epidemic describes the population level epidemiological state while observed data of reported cases are modeled through a separate stochastic observation process. The model distinguishes explicitly between epidemic process stochasticity and reporting stochasticity. 

The result is a stochastic MSEIR state space model where the latent epidemic produces discrete incidence events that are then converted to reported cases through a negative binomial observation model. This formulation is a better modeling approach than a classical deterministic ODE of epidemic trajectory since case observations are noisy and incomplete. The statistical objective of this stochastic model becomes integration over the unobserved epidemic histories rather than selecting a single deterministic trajectory that minimizes residuals. 

## MSEIR State Space Model and Likelihood Inference

### Latent Epidemic
At time $t$ we define the epidemic state as 

$$X(t) = (M_a, S_a, E_a, I_a, R_a)_{a=1}^6$$

which is unobserved with

$$M_a, S_a, E_a, I_a, R_a \in \mathbb{Z}_{\geq 0}$$

We define the observations as the reported measles cases

$$Y_1, \dots, Y_T$$

Which do not match $X(t)$ since the number of reported cases does not equal the number of infections.

### Continuous Time Markov Chain

We model the epidemic as a continuous time Markov chain (CTMC). The memoryless property provides that the future state is conditional on the current state and independent of all previous states. This process is defined by a collection of $j$ transition events characterized by the state change vector $v_j$ and state and time depenedent intensity $a_j(X, t; \theta)$ for model parameters $\theta$. When $j$ occurs,

$$X \to X+v_j$$

Consider when an individual transitions from exposed to infected

$$v_{E \to I, a} = -e_{E_a} + e_{I_a}$$

for unit vectors $e$ denoting state coordinates. The complete epidemic model is therefore defined by a collection 

$$M_\theta = \{X, v_j, a_j\}$$

and the simulation is a numerical process for generating realizations of this model.

### Transition Intensities
For an event of type $j$, the transition intensity is

$$P(\text{one event of type j occurs in } [t, t+dt] \mid X(t) = x) = a_j(X,t;\theta)dt + o(dt)$$

meaning intensity is an instantaneous hazard rate rather than a deterministic flow. Consider the exposed to infection transition, which for incubation rate $\sigma$ has intensity

$$a_{E \to I, a}(X,t) = \sigma E_a$$

so that

$$P(E_a \to I_a \text{ during } [t, t+dt]) = \sigma E_adt + o(dt)$$

We represent maternal immunity loss $\delta$, recovery rate $\gamma$, birth rate, death rate, aging, and vaccination $v$ through the same process. We can represent the expected rate of change using the infinetsmal generator of the CTMC for a suitable function $f$

$$\mathcal{L}_tf(x) = \sum_j a_j(X,t;\theta)(f(x+v_j) - f(x))$$

### Transition Structure
For each age group $a$ the principle transitions are

$$M_a \to S_a, \quad S_a \to E_a, \quad E_a \to I_a, \quad I_a \to R_a$$

with intensities

$$\begin{aligned}
a_{M_a \to S_a,\ a} &= \delta M_a \\
a_{S_a \to E_a,\ a} &= \lambda_a(t)S_a \\
a_{E_a \to I_a,\ a} &= \sigma E_a \\
a_{I_a \to R_a,\ a} &= \gamma I_a
\end{aligned}$$

also included are births, deaths, again, and vaccination. We model vaccination as $S_a \to R_{a+1}$ at a fraction $v_a$ and $S_a \to S_{a+1}$ at $1-v_a$.

### Force of Infection and Seasonal Transmission
The force of infection for age group $a$ is

$$\lambda_a(t) = \beta(t)\sum_bC_{ab}\frac{I_b}{N_b}$$

where $C$ is the contact matrix. Seasonal transmission is modeled by

$$\beta(t) = \beta_0\big(1+\beta_1cos(2\pi t + \phi)\big)$$

The contact matrix is normalized to unit spectral radius

$$\rho(C) = 1 $$

to allow a convenient $R_0$ estimate

$$R_0 = \frac{\beta_0}{\gamma}\rho(C) = \frac{\beta_0}{\gamma}$$

### Incidence

Incidence is defined from realized transition events. Let

$$K_{E \to I, a}(t)$$

be the count of the cumulative number of transitions $E_a \to I_a$ up to time $t$. Therefore the incidence from group $a$ during the observation interval $t$ is

$$C_{a, t} = K_{E \to I, a}(t_t) - K_{E \to I, a}(t_{t-1})$$

and the total incidence is

$$C_t = \sum_a C_{a,t}$$

$C_t$ is a realized stochastic event count. In contrast, the integral

$$\int_{t-1}^t \sigma E_a(s)\ ds$$

represents the expected number of $E_a \to I_a$ events conditional on latent trajectory.

### Observation Model
The realized incidence $C_t$ is not observed directly. Reported cases are modeled by a negative binomial distribution

$$ Y_t \mid C_t \sim \text{NegBin}(\mu_t,\phi_{\text{obs}})$$

with $\mu_t = \rho C_t$ for reporting probability $\rho$. $\phi_{\text{obs}}$ controls the observation overdispursion. The conditional variance is

$$\text{Var}(Y_t \mid C_t) = \mu_t + \frac{\mu_t^2}{\phi_{\text{obs}}}$$

representing both incomplete reporting and extra Poisson variability.

Process noise and observation noise are treated as distinct sources of variability. Process noise is from stochastic transitions and determine $X(t)$ while observation noise is from reporting and determines $Y_t \mid C_t$. We assume observations are independent across intervals conditional on latent incidence counts so the conditional distribution is 

$$p_\theta(Y_{1:T} \mid X_{0:T}) = \prod_{t=1}^Tp_\theta(Y_{t} \mid C_{t})$$

### Joint Probability Model
Let $X_{0:T}$ denote the latent epidemic trajectory and $Y_{1:T}$ the observed cases. The joint probability model is

$$p_\theta(X_{0:T}, Y_{1:T}) = p_\theta(x_0)p_\theta(x_{0:T} \mid x_0)\prod_{t=1}^Tp_\theta(Y_{t} \mid C_{t})$$

The first term specifies the initial state distribution, the second is induced by the Markov chain, and the final product is the observation model. Since the epidemic trajectory is unobserved, inference is based on the marginal likelihood

$$\mathcal{L}(\theta) = p_\theta(Y_{1:T}) = \int p_\theta(x_0)p_\theta(x_{0:T} \mid x_0)\prod_{t=1}^Tp_\theta(Y_{t} \mid C_{t}) \ dX_{0:T}$$
which averages the probability of observations over all trajectories in the stochastic model.

### Particle Filtering
A particle filter is used to estimate the hidden state of the CTMC using noisy observations. The latent state filtering distribution of the model

$$p(X_t \mid Y_{1:T}, \theta)$$

is analytically intractable for the nonlinear and high dimensional model, so we approximate it using a particle filter. For each observation time, the $N_p$ particles

$$X_t^{(1)}, \dots X_t^{(N_p)}$$

represent plausible latent epidemic histories. Each particle is propagated through the stochastic process and the incidence is recorded. Particles are then weighted using observation likelihood

$$w_t^{(i)} = p_\theta(Y_t \mid C_t^{(i)})$$

Applying the negative binomial observation model

$$w_t^{(i)} = \text{NegBin}(Y_t ; \rho C_t^{(i)},\phi_{\text{obs}})$$

and normalizing

$$\tilde{w}_t^{(i)} = \frac{w_t^{(i)}}{\sum_{k=1}^{N_p} w_t^{(k)}} $$

Particles are resampled according to the normalized weights and trajectories that make the observations probable receive greater representation in the subsequent population. The particle filter can be interpreted as Monte Carlo integration over the trajectories in the marginal likelihood. 

### Particle Likelihood Estimation
The marginal likelihood can be factored as

$$p_\theta(Y_{1:T}) = \prod_{t=1}^T p_\theta(Y_t \mid Y_{1:t-1})$$

The particle filter estimates each, giving 

$$\hat L(\theta) = \prod_{t=1}^T \hat p_\theta(Y_t \mid Y_{1:t-1})$$

and therefore

$$\widehat{logL}(\theta) = \sum_{t=1}^T\log\hat p_\theta(Y_t \mid Y_{1:t-1})$$

which is stochastic because it depends on random trajectories and resampling. 

### Numerical Simulation
The stochastic model is simulated numerically using an Euler multinomial step algorithm (EM). EM advances the epideic state over a fixed time interval $\Delta t$ by jointly sampling the number of individuals under going each transitiion. For a state transition with rate $r_1$, the probability an individual transitions during $\Delta t$ is

$$p = 1-e^{-r \Delta t}$$

Therefore for $n$ individuals in a compartment the number transitioning is sampled from

$$K \sim \text{Binomial}(n, p)$$

Since individuals can transition through muliple mututally exclusive paths, the transitions are sampled jointly using a multinomial distribution. For $J$ possible transition paths with probability $p_j$ the residual probability

$$ p_0 = 1 - \sum_{i=1}^J p_j $$

and thus

$$(K_1, \dots, K_j, K_0) \sim \text{Multinomial}(n;p_1, \dots, p_J, p_0)$$