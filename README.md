# PhyFSME: A Physics-Inspired Multi-Scale Fractional Spectral Mixture-of-Experts for Non-Stationary Time Series Forecasting  

Multivariate time series forecasting (MTSF) is essential for many real-world applications, including
traffic dispatching, energy management, weather warning, and industrial control. However, real-world
multivariate series often contain long-term trends, abrupt local changes, cross-variable couplings, and
time-varying frequency structures, making it challenging for models based on fixed temporal representations or fixed Fourier bases to capture compact and stable non-stationary dynamics. To this end,
we propose PhyFSME, a physics-inspired multi-scale fractional spectral mixture-of-experts (MoE)
framework for non-stationary MTSF. PhyFSME introduces learnable Fractional Fourier Transform
(FRFT) operators into a fractional spectral stream, enabling adaptive time-frequency rotations and
data-dependent spectral projections. It further employs multi-scale FRFT analysis windows with a
Scale-MoE routing mechanism to fuse spectral representations under different temporal supports,
capturing long-term trends, stable periodicity, and local transients. To stabilize complex-domain
optimization, we design a complex-domain consistency regularization that constrains the inverseFRFT representation in terms of imaginary energy ratio, local smoothness, and overall magnitude.
PhyFSME also integrates a variable interaction branch and dynamic gated fusion to jointly model
cross-variable dependencies and non-stationary spectral dynamics. Extensive experiments on 14
benchmark and real-world datasets demonstrate that PhyFSME achieves consistent advantages in
forecasting accuracy, structural stability, and robustness, validating the effectiveness of adaptive
fractional time-frequency modeling for non-stationary MTSF.
