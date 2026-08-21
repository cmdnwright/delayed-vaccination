## An Index of Notable Figures Produced During the Research Process

- **gradient_descent_loss** — Contains optimization metrics for initial gradient descent approach to model fitting. Gradient norm led to considering model scaling; parameter trajectories led to considering fixing biological parameters.

- **deterministic_dampened_oscilator.png** — Observed vs. modeled cases after adding a burn-in period to reduce initial peaks. Dampened oscillator behavior led to addition of seasonality parameters in transmission.

- **deterministic_seasonal_transmission.png** — Observed vs. modeled cases after adding seasonal transmission to the model. Magnitude scale led to reporting rate scaling, and still non-biannual cycles led to further investigation of periodicity.

- **deterministic_reporting_scaling.png** — Observed vs. modeled cases after adding a scalar to account for underreporting. Lack of biannual cycles led to variable vital rates, since birth and death rates drive periodicity.

- **variable_vital.png** — Observed vs. modeled cases after introducing time-dependent functions for births and deaths. Continued periodicity mismatch led to bifurcation analysis.

- **deterministic_bifurcation.png** — Bifurcation grid on birth rate and beta one of deterministic ODE model, checking for biannual cycles. Lack of biannual cycles across two main drivers of periodicity surfaced solver instability during further analysis.

- **deterministic_LSODA.png** — Observed vs. modeled cases after changing the ODE solver. Investigation of solver instability led to using LSODA instead of traditional RK4 methods to improve performance on stiff regions, which altered model behavior. Further investigation revealed continued instability in the burn-up period specifically, despite the enhanced solver, ultimately pushing toward the non-deterministic models.

- **ess_zero.png** — Log likelihood and effective sample size over IF2 fitting. ESS of zero is degenerate, and combined with a standard deviation analysis, revealed a flaw in the number of particles chosen per simulation that was leading to frequent extinction and exploding Monte Carlo error. That analysis ultimately surfaced an error in the IF2 implementation (non-sequential evaluations for all thetas) that was further contributing to ESS problems and runtime inflation.