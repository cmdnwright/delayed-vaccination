# Simulating Measles Cases Under Plausible Counterfactual Vaccine Scenarios

We use continuous time Markov chains to model unseen latent measles epidemic trajectories under a partially observed stochastic state space framework with observed cases modeled through a negative binomial observation process. The model is calibrated to pre-vaccination era data using a particle filter to estimate the hidden state and an Euler Multinomial step algorithm for numerical simulation. The model is then validated using historical vaccination and used for experiments on counterfactual vaccine scenarios in a post extinction measles regime.

### Key Findings

Maintaining high levels of vaccination at the CDC recommended age keeps cases at a low baseline driven mostly by stochastic importations. Delaying vaccination later than the CDC recommended schedule for the majority of children, in contrast, leads to persistent transmission and endemic equilibrium dynamics in both delayed vaccination scenarios. A Holm-Bonferroni corrected paired sign flip permutation test comparing both delayed vaccination scenarios to continued high, recommended coverage finds statistically significant differences in the mean cumulative cases over a five year simulation period.

## Motivation

Disease epidemics are inherently stochastic and noisy. Disease dynamics are discrete transitions between epidemiological states and can be naturally represented using state dependent hazard rates. For a historically pervasive and infectious disease like measles, case underreporting also adds a layer of probabilistic complexity to the model. As a result, modeling measles is both computationally interesting and an important predictive tool. Understanding epidemic dynamics can allow us to anticipate both significant macro level events and the subsequent industry and policy responses. In 2025, the United States saw its largest surge in measles cases since the disease was deemed eradicated in 2000. Low vaccination rates among populations in Texas specifically created the necessary conditions for an outbreak to spark. This project seeks to use a stochastic MSEIR measles model with particle filtering and Euler Multinomial step numerics to simulate how counterfactual vaccination rate scenarios could affect measles dynamics in Texas as a framework for anticipating potential dynamic shifts for various future scenarios of the MMR vaccine.

## Model

We attempt to capture the most accurate representation of measles dynamics using a standard stochastic MSEIR model. The maternal immunity and latent exposed periods both reflect key biological components of the measles virus that are primary drivers of its periodicity and dynamics. Our model also incorporates open demography with time dependent birth and death rates interpolated from census data to further drive dynamics and preserve the population scale accuracy over 75 years of forward simulation. The final standard component of modern measles models also implemented here is seasonal forcing on transmission, though we use classical cosine forcing rather than modern step forcing mirroring school schedules. 

Well documented epidemiological parameters including recovery rate, maternal immunity waning rate, and transition rate are fixed from literature to preserve identifiability. The remaining parameters fit by the IF2 iterated fitting algorithm on the 1950-1965 pre-vaccine calibration window are the average transmission, seasonal amplitude, seasonal phase, reporting rate, and over dispersion of the negative binomial observation model. 

See [`METHODS.md`](METHODS.md) for a full mathematical formulation of the simulation and reporting models.

## Experiments

We run three primary counterfactual experiments on the vaccination rates. All are forward simulations from the end of the validation window, starting in 2000 when both historical data and our model show measles to be eradicated. The age stratified structure of our model allows us to vary the vaccination rate per age and observe the resulting simulation.

The first simulation sustains the high levels of vaccine coverage that ultimately contributed to the eradication of measles. We set the 5 year old vaccination rate to 97% to be consistent with the hard vaccination rate for all public schools, and the 1 year old rate at 92% in line with the majority of people following the CDC recommended schedule. We treat this simulation as the baseline. 

The key scenario of interest shifts the majority of vaccination before school age from 2 years old as recommended by the CDC to 3 years old, following some common delayed schedules. School age vaccination is left at 97%.

Finally, we simulate an extreme where vaccination only occurs at school age. 

We calculate the mean cumulative cases at 2000 particles over 25 years of simulation for all scenarios and compare the two delayed vaccination schedules to the high coverage baseline. Since all simulations are run with the same initial cloud and random seed, particle $i$ in both simulations shares the same random draws up to the vaccination treatment, so we pair particles for the paired sign flip permutation test. Holm-Bonferroni correction is applied to reduce false positive rates from repeated statistical tests.

**Simulation Metrics**
| Scenario | Mean Cumulative Cases | p-value vs Baseline | Holm Corrected |
|---|---|---|---|
| Baseline | 125.2585 | - | - |
| Delayed | 2697.781 | 1.0e-5 | 2.0e-5|
| School Age Only | 572617.7665 | 1.0e-5 | 2.0e-5 |

*The paired sign flip permutation test uses Monte Carlo resampling to approximate the null distribution at 200,000 samples. The minimum p-value is thus* $\frac{2}{n_\text{resamples} + 1} = 9.99995 \times 10^{-6}$ *which is the exact p value reported by both tests. We round that number to* $1.0 \times 10^{-5}$ *for the sake of legibility.*

We also perform a sensitivity sweep over the 3 year old vaccination rate, forward simulating and reporting the mean cumulative cases and 95-5% interval as a function of the vaccination rate.

## Results

-**Calibration successfully converges to reasonable dynamics with numerical stability and low ESS extinction rates.** The original deterministic model struggled with both matching the well known disease dynamics and remaining stable during optimization under the large population size but small magnitude of cases. Nearly every deterministic solve ran into stability issues with case explosions, even during a burn in period, that corrupted the nonlinear least squares optimizer. Initial iterations of the stochastic model suffered from high ESS extinction rates, which are also addressed by the current model using a spin up period and the IF2 algorithm.

-**The model successfully achieves extinction dynamics after the vaccine ramp up during validation.** While the model consistently overpredicts cases during the validation window, and notably predicts the 1990 surge earlier than the data, the overall peak timing of the model is consistent with the data. The errors are primarily in magnitude during peak years which could be attributed to sensitive or historically inaccurate fixed parameters like the recovery or historical vaccination uptake. Most importantly, by the end of the calibration window the model has successfully achieved extinction dynamics.

-**Delayed vaccination of any kind as a statistically significant impact on cases.** While the baseline high coverage scenario maintains cases at the level only of noisy importation, both delayed scenarios produce an increase in cases that is statistically significant when compared to the baseline though mean cumulative case difference. 

-**A sensitivity sweep on the toddler vaccination rate under the delayed schedule framework reveals case increases at all levels of coverage.** Even at modern coverage levels of uptake, any simulation where the majority of children are vaccinated at age 3 rather than age 2 produces nontrivial case counts. 

## Limitations and future work

- Historical age stratified disease data of case counts broken down by desired age range is difficult to find. While the age stratified formulation of this model could allow for age level analysis on case dynamics, the lack of this data means the model cannot be validated at this resolution and thus that level of analysis was left out.

- The contact matrix used for this project was binned in 5 year age gaps. While this was used to estimate contacts at the desired level of age stratification, the approach used is an averaging process which assumes contacts within the bin are uniformly distributed across the age range, missing out on real nuance between age ranges. Higher resolution contacts would enable further age specific analysis.

- Seasonal forcing is a key driver of periodicity for measles models. This model implements seasonal forcing using a classic cosine forcing term, but modern models use step forcing functions that can more accurately reflect school terms. Recent research demonstrates that long term epidemic bifurcations are invariant to the shape of seasonal forcing, especially since the majority of the models compartments exist before the school age and the school age population is completely integrated into the rest of the adult population (Papst and Earn 2019).

- The model fits the reporting rate as a constant parameter. Historical results show that the true reporting rate has varied over time, especially at the individual state level. While fitting this parameter allows the model to mitigate some of those effects, a time dependent reporting rate could improve the observation model, though potentially at the cost of identifiability.

- This model treats the measles and later MMR vaccine as perfectly effective and as a single dose with no waning immunity. The modern MMR vaccine is a two dose vaccine, which could be better modeled by discriminating between doses through the introduction of more compartments for first dose and second dose as well as waning immunity in the first dose compartments rather than permanent immunity after vaccination.



## Repository structure

```
delayed-vaccination/
|-- data/
|   |-- raw/ # raw data of cases, populations, and contact matrix
|   |-- processed/ # rebinned contact matrix, case time series, and calibration/validation results
|-- scripts/ # calibration and validation scripts
|-- src/
|   |-- calibration/ # calibration modules and metrics
|   |-- data/ # data processing modules
|   |-- experiments/ # counterfactual scenarios and vaccination framework
|   |-- model/ # stochastic MSEIR, particle filter, and numerical simulation
|   |-- validation/ # validation modules and metrics
|   |-- config.yaml # config for model and experimental constants 
|-- tests/ # data and model integrity tests
|-- model_calibration.ipynb # calibration check
|-- model_validation.ipynb # validation metrics
|-- model_experiments.ipynb # experimental results
```
All scripts and modules are fully documented. See source files for implementation decisions.

## Reproducing this project

```bash
# 1. clone and install
git clone <repo-url>
pip install -r requirements.txt

# 2. place tycho and NHGIS data and contact matrix in data/raw/

# 3. run calibration and validation scripts
python -m scripts.run_calibration
python -m scripts.run_validation

# 4. run all notebooks
```

Calibration and validation scripts took less than 10 minutes each using up to 8 cores parallelized across particle filter workers. Validation and calibration notebooks require about 20 minutes each for specific plots running on 10 cores. Each experimental simulation takes about 2 minutes on a single core and the sensitivity sweep takes about 20 minutes.

## Data
[Project Tycho](https://www.tycho.pitt.edu/)
- Texas measles case series

[IPMUS NGHIS population data](https://www.nhgis.org/)
- Population by age filtered for texas in 2020

Fixed epidemiological parameters set using CDC reports, local public health administration data, and peer-reviewed literature. 
