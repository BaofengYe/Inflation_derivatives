# Assessing Seasonality in Inflation-Linked Derivatives

### A Framework for Detection, Estimation, and Pricing-Consistent Modelling of the Seasonal Component of Consumer Price Indices

***

**Abstract.** Consumer price indices exhibit strong, largely predictable intra-annual variation that must be modelled explicitly when valuing inflation-linked derivatives, because these instruments reference the _non-seasonally-adjusted_ (NSA) index whereas statistical agencies publish and revise a _seasonally-adjusted_ (SA) counterpart. This paper formalises the problem. The framework (i) fixes the decomposition and the no-arbitrage normalisation that any admissible seasonal estimate must satisfy; (ii) identifies where the seasonal component enters the valuation of zero-coupon and year-on-year inflation swaps and of inflation-linked bonds through the daily reference index and the indexation lag; (iii) catalogues the statistical tools for _detecting_ seasonality, for classifying it as _deterministic versus stochastic_ (seasonal unit-root tests), and for testing its _temporal stability_; (iv) presents the competing estimation models — deterministic factor vectors, harmonic regressions, `regARIMA`/X-13ARIMA-SEATS, TRAMO-SEATS, STL, structural (state-space) models, SARIMA, bottom-up component aggregation, and recent machine-learning approaches — with their mathematical specifications and trade-offs; and (v) addresses the practical questions that dominate profit-and-loss: the COVID-era structural break and outlier treatment, index-specific conventions, market-implied seasonality as a relative-value signal, and the quantification of seasonal model risk.

**Keywords:** inflation derivatives, seasonal adjustment, zero-coupon inflation swap, HEGY test, Canova–Hansen test, X-13ARIMA-SEATS, structural time series, Kalman filter, indexation lag, model risk.

***

## 1. Introduction and scope

Inflation-linked instruments — zero-coupon inflation-indexed swaps (ZCIIS), year-on-year inflation-indexed swaps (YYIIS), inflation caps/floors, and inflation-linked bonds (ILBs, "linkers") — settle against a published price index. The three benchmark references are the U.S. CPI-U (all urban consumers), the Euro-area HICP ex-tobacco (HICPxT), and the U.K. RPI. In every case the instrument references the **NSA** index. This institutional convention gives rise to the core modelling problem: the analyst must reconstruct the seasonal component from published data in order to (a) obtain the expected NSA fixing at _any_ reference month (including non-standard months implied by an ILB's dated cashflows), and (b) impose that component on a forward curve calibrated to a discrete set of liquid quotes without disturbing the calibration.

The seasonal effect is not a second-order refinement at the front end. It is the primary driver of pricing differences at short maturities and for the terminal reference month of a zero-coupon structure, and it governs the "carry over a print" — the deterministic index accrual across a single monthly CPI release. Seasonal effects largely cancel in year-on-year structures (a given calendar month is compared against the same calendar month a year earlier) but do not cancel exactly once seasonality is allowed to evolve.

The remainder of this note is organised as: notation and decomposition (§2); the normalisation constraint (§3); valuation entry points (§4); detection, classification, and stability tests (§5–§7); estimation models (§8); pricing-consistent curve construction (§9); estimation pitfalls including the pandemic break (§10); market-implied seasonality (§11); model-risk quantification (§12); and a recommended workflow (§13). Appendices give the explicit quarterly HEGY regression and the state-space system matrices for the structural model.

***

## 2. Notation, the observation object, and the decomposition

Let $\{I_t\}_{t\in\mathbb{Z}}$ denote the _level_ of the NSA index observed at monthly frequency, and $y_t \equiv \log I_t$. The classical multiplicative decomposition in levels, and its log-additive form, are

​$I_t \;=\; \mathrm{TC}_t \cdot S_t \cdot \varepsilon_t, \qquad\Longleftrightarrow\qquad y_t \;=\; \tau_t + s_t + \iota_t,$​

where $\mathrm{TC}_t=e^{\tau_t}$ is the **trend-cycle** (the smooth, seasonally-adjusted signal), $S_t=e^{s_t}$ is the **seasonal** factor, and $\varepsilon_t = e^{\iota_t}$ is the **irregular** (calendar and outlier effects are usually separated out of the irregular into a distinct component, see §8.3). Working in logs, month-on-month (MoM) and year-on-year (YoY) log-inflation are

​$\pi^{\text{MoM}}_t = y_t - y_{t-1} = \Delta y_t, \qquad \pi^{\text{YoY}}_t = y_t - y_{t-12} = \Delta_{12} y_t .$​

Because $\Delta_{12} = \Delta \, U(L)$ with $U(L)=1+L+\cdots+L^{11}$, the annual difference sweeps a full seasonal cycle, which explains why YoY quantities are first-order seasonality-free.

The publication convention matters: the index for reference month $m$ is released at $m{+}1$ (mid-month for U.S. CPI), inducing a natural information lag that combines with the contractual **indexation lag** $L_{\text{idx}}$ (§4.3). U.S. CPI-U NSA carries the property that it is _not revised_ after first publication; HICPxT is revisable; agency-published _seasonal factors_ are, in all cases, revised on an annual schedule.

***

## 3. No-arbitrage normalisation of the seasonal component

Any admissible seasonal estimate must be **normalised so that it contributes nothing to inflation measured over a full year**. In log space,

​$\boxed{\;\sum_{j=0}^{11} s_{t-j} \;=\; 0 \quad\text{for every } t\;} \qquad\Longleftrightarrow\qquad \prod_{j=0}^{11} S_{t-j} \;=\; 1 .$​

For a _deterministic_ twelve-factor model with time-invariant factors $\gamma_1,\dots,\gamma_{12}$ (indexed by calendar month), (3.1) reduces to the single linear restriction $\sum_{m=1}^{12}\gamma_m=0$. For _time-varying_ seasonality, (3.1) must hold over **every rolling twelve-month window**, not merely the calendar year January–December; otherwise a spurious low-frequency drift leaks into the trend and the model injects phantom annual inflation.

The economic interpretation of (3.1) is a no-arbitrage requirement in the following operational sense. The forward index curve is calibrated to liquid ZCIIS quotes, which are struck at (predominantly) annual pillars. If the seasonal factors integrate to zero over each rolling year, they vanish at those annual pillars and therefore leave the fitted annual breakevens — hence the repricing of the calibrating instruments — unchanged. A seasonal vector violating (3.1) either mis-reprices the calibration set or manufactures carry that is not present in traded prices. This is the sense in which seasonality must be layered on _arbitrage-free_.

***

## 4. Where seasonality enters valuation

### 4.1 Zero-coupon inflation-indexed swap (ZCIIS)

A ZCIIS maturing at $T_M$ ($M$ integer years) exchanges, at $T_M$, the realised index return against a compounded fixed rate $K$:

​$\text{floating: } \frac{I_{T_M}}{I_{0}} - 1, \qquad \text{fixed: } (1+K)^{M} - 1 .$​

The ZCIIS admits a _model-independent_ valuation (Mercurio, 2005; Brigo & Mercurio, 2006). Working under the nominal $T_M$-forward measure $Q_n^{T_M}$ and invoking the foreign-currency analogy of Jarrow & Yildirim (2003) — treating the CPI as the exchange rate between a "nominal" and a "real" economy — the expected terminal index is

​$\mathbb{E}^{Q_n^{T_M}}\!\big[I_{T_M}\big] \;=\; I_0\,\frac{P_r(0,T_M)}{P_n(0,T_M)} \;\equiv\; \mathcal{F}_I(0,T_M),$​

where $P_n(0,T)$ and $P_r(0,T)$ are nominal and real zero-coupon bond prices. Equation (4.1) _defines_ the market-implied forward index $\mathcal{F}_I(0,T_M)$, and the par fixed rate solves $(1+K(T_M))^M = \mathcal{F}_I(0,T_M)/I_0$.

The seasonal component enters when the forward index is required at a **non-pillar reference month** $t$. One decomposes the (log) forward into a smooth deseasonalised trend $\hat\tau_t$ and the seasonal factor,

​$\mathcal{F}_I(0,t) \;=\; \exp\!\big(\hat\tau_t + \hat s_t\big), \qquad \hat s_t = \hat s_{\,\mathrm{month}(t)} ,$​

with $\hat\tau_t$ interpolated between the (deseasonalised) annual pillars and $\hat s_t$ supplied by the seasonal model. Normalisation (3.1) guarantees that (4.2) collapses to the calibrated value at the annual pillars.

### 4.2 Year-on-year inflation-indexed swap (YYIIS)

A YYIIS pays, at each $T_i$, the annual index return $I_{T_i}/I_{T_{i-1}} - 1$ against a fixed rate. Unlike the ZCIIS, the YYIIS forward is **model-dependent**: it requires a convexity adjustment because $\mathbb{E}[I_{T_i}/I_{T_{i-1}}] \neq \mathbb{E}[I_{T_i}]/\mathbb{E}[I_{T_{i-1}}]$ in general. Schematically, with the forward index decomposed as in (4.2),

​$\mathbb{E}^{Q_n^{T_i}}\!\left[\frac{I_{T_i}}{I_{T_{i-1}}}\right] \;\approx\; \underbrace{\frac{e^{\tau_{T_i}}}{e^{\tau_{T_{i-1}}}}}_{\text{trend YoY}} \cdot \underbrace{\frac{S_{T_i}}{S_{T_{i-1}}}}_{\text{seasonal ratio}} \cdot \underbrace{\mathcal{C}(T_{i-1},T_i)}_{\text{convexity}} ,$​

where $\mathcal{C}$ depends on the assumed dynamics (e.g. a Jarrow–Yildirim or market-model specification with nominal/real rate and index volatilities and correlations). If $T_i$ and $T_{i-1}$ are the _same calendar month_ one year apart and seasonality is time-invariant, then $S_{T_i}/S_{T_{i-1}}=1$ exactly — the first-order cancellation. Under evolving seasonality the ratio departs from unity by the amount of intra-year seasonal drift, which is the second-order effect that distinguishes competing models at the YoY level.

### 4.3 Inflation-linked bonds: daily reference index and indexation lag

For ILBs and asset swaps thereon, coupons and redemption accrete by a **daily reference index** (DRI) obtained by linear interpolation between two monthly fixings, lagged by $L_{\text{idx}}$ months:

​$\mathrm{DRI}(d) \;=\; I_{\,m - L_{\text{idx}}} \;+\; \frac{d-1}{D_m}\,\Big( I_{\,m - L_{\text{idx}} + 1} - I_{\,m - L_{\text{idx}}} \Big),$​

where $d$ is the day within settlement month $m$ and $D_m$ the number of days in that month. Standard lags are $L_{\text{idx}}=3$ months (U.S. TIPS, French OAT€i, and modern "3-month" linkers) and $L_{\text{idx}}=8$ months (legacy U.K. RPI linkers). Two consequences for seasonality:

1. The seasonal factor that applies to a given cashflow is that of the **reference month** $m-L_{\text{idx}}$, not the payment month $m$.
2. Seasonality resides entirely in the monthly fixings $I$ and is _inherited_ by the daily interpolation (4.4). A common implementation error is to super-impose an intramonth seasonal shape on top of (4.4), double-counting the effect that the monthly nodes already carry.

Because the front end is where the seasonal factor is largest relative to the trend, dedicated short-dated products — month-by-month "seasonal swaps" — trade precisely to isolate and hedge this exposure, and vendor pricing for the front end carries the seasonal factor explicit (Parameta Solutions, 2025).

***

## 5. Detecting seasonality

Standard practice establishes the presence of identifiable seasonality before estimation. Three complementary tests are widely used.

### 5.1 Parametric $F$-test on seasonal dummies

Regress MoM log-inflation on eleven monthly dummies (plus a constant, and optionally a deterministic trend):

​$\Delta y_t \;=\; \alpha + \sum_{m=2}^{12}\gamma_m D_{m,t} + u_t, \qquad H_0:\;\gamma_2=\cdots=\gamma_{12}=0 .$​

The joint restriction is tested with a standard $F$-statistic (HAC-robust standard errors are advisable given autocorrelated $u_t$). This first-pass check assumes _stable_ seasonality and is not robust to a small number of extreme months.

### 5.2 Kruskal–Wallis test for stable seasonality (non-parametric)

Pooling the seasonal-irregular ratios (or MoM changes) across $N$ observations and ranking them, with $R_m$ the rank sum and $n_m$ the count for calendar month $m$,

​$H \;=\; \frac{12}{N(N+1)}\sum_{m=1}^{12}\frac{R_m^{2}}{n_m} \;-\; 3(N+1) \;\;\overset{H_0}{\sim}\;\; \chi^2_{11},$​

under the null of no stable seasonality. This is the rank-based test embedded in the X-11/X-13 diagnostic suite; being distribution-free it is robust to non-normal irregulars.

### 5.3 The $QS$ statistic (residual seasonality)

A one-sided Ljung–Box variant restricted to the first two seasonal lags, counting only _positive_ autocorrelations $\hat\rho(12k)$ (negative estimates set to zero, since genuine seasonality induces positive seasonal autocorrelation):

​$QS \;=\; T(T+2)\sum_{k=1}^{2}\frac{\big[\max(0,\hat\rho(12k))\big]^{2}}{T-12k} \;\;\overset{H_0}{\sim}\;\; \chi^2_{2}.$​

​$QS$ is the standard residual-seasonality diagnostic in TRAMO-SEATS / JDemetra+ and is applied both to the raw series (to confirm seasonality) and to the adjusted series and irregular (to confirm its _absence_ post-adjustment).

### 5.4 Spectral diagnostics

Estimate the (AR- or periodogram-based) spectral density $\hat f(\omega)$ and inspect for peaks at the six monthly seasonal frequencies $\omega_k = 2\pi k/12,\ k=1,\dots,6$, and at trading-day frequencies. A correctly adjusted series exhibits _no_ residual peaks at these frequencies; the visual "Tukey" spectrum plot in X-13 formalises this check.

### 5.5 The combined X-11/X-13 seasonality test ($M7$)

X-11 lineage decomposes variance in the seasonal-irregular (SI) ratios via a two-way analysis of variance into a **stable-seasonality** $F$-statistic $F_S$ (between-months mean square over residual mean square, with $(11,\,N-12)$ degrees of freedom) and a **moving-seasonality** $F$-statistic $F_M$ (a year effect). The Lothian–Morry combined statistic is

​$M7 \;=\; \sqrt{\tfrac12\left(\frac{7}{F_S} + \frac{3\,F_M}{F_S}\right)}, \qquad \text{identifiable seasonality} \iff M7 < 1 .$​

​$M7$ jointly rewards strong stable seasonality (large $F_S$) and penalises seasonality that moves too fast to be reliably estimated (large $F_M$).

***

## 6. Deterministic versus stochastic seasonality

Whether the seasonal pattern is **deterministic** (fixed, or slowly and smoothly evolving) or **stochastic** (integrated at seasonal frequencies, i.e. carrying seasonal unit roots) is the key modelling decision: a fixed factor vector applied to a series with genuine seasonal unit roots will drift out of sample. The two canonical tests have _opposite_ nulls and are used jointly (Hylleberg et al., 1990; Canova & Hansen, 1995; Franses, 1996).

### 6.1 The seasonal difference operator and its roots

For monthly data the seasonal difference factors over the reals as

​$1 - L^{12} = (1-L)(1+L)(1+L^2)(1+L+L^2)(1-L+L^2)(1+\sqrt3\,L+L^2)(1-\sqrt3\,L+L^2),$​

whose roots are the twelve roots of unity, i.e. the frequencies

​$\omega \in \Big\{\,0,\; \tfrac{\pi}{6},\; \tfrac{\pi}{3},\; \tfrac{\pi}{2},\; \tfrac{2\pi}{3},\; \tfrac{5\pi}{6},\; \pi \,\Big\},$​

with $\omega=0$ the (non-seasonal) zero frequency and the remaining six the seasonal frequencies (complex ones entering in conjugate pairs). Testing for a seasonal unit root means testing whether the corresponding factor is present in the autoregressive representation.

### 6.2 HEGY test — null of a seasonal unit root

Hylleberg, Engle, Granger & Yoo (1990) devise auxiliary variables that isolate each frequency and run a single regression whose coefficients switch each unit root on or off. In the monthly extension (Franses, 1991; Beaulieu & Miron, 1993) the regression is

​$\varphi(L)\,\Delta_{12}y_t = \sum_{k=1}^{12}\pi_k\, z_{k,t-1} + \mu_t + e_t,$​

where $\mu_t$ collects deterministic terms (constant, trend, seasonal dummies), $\varphi(L)$ augments with lags of $\Delta_{12}y_t$ to whiten $e_t$, and $\{z_{k,t}\}$ are the frequency-isolating filters. Hypotheses (HEGY convention): $\pi_1=0$ tests the zero-frequency unit root ($t$-test, one-sided); $\pi_2=0$ tests the root at $\omega=\pi$; and each complex seasonal frequency is tested by the **joint** nullity of its pair, e.g. $\pi_3=\pi_4=0$, via an $F$-test. The overall null of "seasonal integration" is that the relevant $\pi$'s are zero; **rejecting** unit roots at all seasonal frequencies, together with significant seasonal dummies, licenses a _deterministic_ seasonal model. The critical values are non-standard (Dickey–Fuller family) and depend on the deterministic terms and sample size; see the explicit quarterly regression in Appendix A.

### 6.3 Canova–Hansen test — null of deterministic (stable) seasonality

Canova & Hansen (1995) reverse the null: they take stationarity around a deterministic seasonal pattern as $H_0$ and test it against the alternative that the seasonal coefficients follow a random walk (i.e. a seasonal unit root under $H_1$). This generalises the KPSS (Kwiatkowski et al., 1992) idea to seasonal frequencies. Writing the seasonal mean via frequency regressors and letting $\hat f_t$ be the score contributions and $\hat F_t=\sum_{i\le t}\hat f_i$ their partial sums, the statistic is a Cramér–von Mises / Lagrange-multiplier form

​$\mathcal{L} \;=\; \frac{1}{T^{2}} \sum_{t=1}^{T} \hat F_t'\, \hat\Omega^{-1}\, \hat F_t,$​

with $\hat\Omega$ a heteroskedasticity-and-autocorrelation-consistent (long-run) covariance estimate. Large $\mathcal{L}$ **rejects** stability, indicating time-varying / stochastic seasonality; the test can be applied frequency-by-frequency or jointly, with asymptotic critical values from the generalised von Mises distribution.

### 6.4 Using HEGY and Canova–Hansen together

The tests are complementary because their nulls are swapped (unit root vs. stationarity). The practical classification is:

| **HEGY (null: unit root)** | **Canova–Hansen (null: deterministic)** | **Conclusion**                                                      |
| -------------------------- | --------------------------------------- | ------------------------------------------------------------------- |
| Reject                     | Do not reject                           | **Deterministic** seasonality — fixed factor vector adequate        |
| Do not reject              | Reject                                  | **Stochastic** seasonality — seasonal differencing / evolving model |
| Do not reject              | Reject at some frequencies              | **Mixed** — deterministic at some frequencies, integrated at others |
| Reject                     | Reject                                  | Ambiguous; favour an evolving (state-space) specification           |

The Osborn–Chui–Smith–Birchenhall (OCSB, 1988) test complements these by assessing whether the series requires _both_ first and seasonal differencing $(1-L)(1-L^{12})$. Empirically, headline CPI series across the OECD test as _predominantly deterministic_ but with documented, non-systematic changes in the seasonal pattern since 1980 (Arend et al., 2024) — which argues against a permanently frozen vector and in favour of periodic re-estimation or an explicitly evolving model (§8.6).

***

## 7. Testing stability and structural change

Even where seasonality is "deterministic" at each date, the _level_ of the factors can shift across regimes (re-weighting of the index basket, changing retail/e-commerce patterns, energy-mix shifts, tax-timing changes, methodology revisions). The relevant diagnostics:

* **Moving-seasonality** $F$**-test** (the $F_M$ entering (5.5)): a formal test that seasonal factors change across years.
* **Canova–Hansen (6.3)** doubles as a stability test — its alternative _is_ time variation.
* **Sliding-spans analysis** (Findley et al., 1998): re-estimate the adjustment over several overlapping spans and flag months whose seasonal factor moves by more than a tolerance (commonly 3%). This is the primary robustness check on an operational adjustment.
* **Revision-history diagnostics**: compare concurrent (first) seasonal factor estimates against final estimates; large revisions indicate an unstable pattern or excessive filter flexibility.
* **Multiple structural breaks**: Bai & Perron (1998, 2003) for endogenous detection of break dates in the seasonal coefficients; Chow tests where the break date is known (e.g. a re-weighting or a methodology change).

A distinct and practically important case — the pandemic — is treated in §10.

***

## 8. Models for the seasonal component

The estimation models below are ordered from market-standard to research-grade.

### 8.1 Deterministic factor vector (the desk standard)

Estimate twelve calendar-month factors and hold them fixed. A standard estimator regresses MoM log-inflation on month dummies as in (5.1), recovers the fitted monthly means, and cumulates and centres them to a factor vector $\{\hat s_m\}_{m=1}^{12}$ satisfying $\sum_m \hat s_m = 0$. Equivalently, one runs an X-13 / TRAMO-SEATS adjustment (§8.3–8.4), extracts the final multiplicative seasonal factors, averages the last $n$ years by calendar month, and renormalises via (3.1). The factor is then applied on top of the deseasonalised forward as in (4.2). Belgrade & Benhamou (2004) review exactly this construction, comparing a parametric least-squares technique with a non-parametric X-11 approach; Kerkhof (2005) sets it in the swap-curve bootstrap. _Advantages_: transparency, ease of hedging, and consistency with market convention (e.g. Bloomberg SWIL). _Limitations_: assumes time-invariant seasonality and is sensitive to the estimation window and to outliers.

### 8.2 Harmonic (trigonometric) regression

Represent the seasonal component as a finite Fourier sum,

​$s_t \;=\; \sum_{k=1}^{K}\Big[\, a_k \cos\!\big(\tfrac{2\pi k t}{12}\big) + b_k \sin\!\big(\tfrac{2\pi k t}{12}\big)\Big], \qquad K \le 6,$​

fitted by ordinary least squares on the detrended series. With $K=6$ (the Nyquist term contributing a single cosine) this reproduces the full eleven-dummy fit; choosing $K<6$ by AIC/BIC yields a _parsimonious, smooth_ factor and acts as a natural regulariser when history is short. Harmonic factors are attractive when a small, differentiable parameterisation is desired for downstream curve construction.

### 8.3 `regARIMA` pre-adjustment and X-13ARIMA-SEATS

The agency-standard method (U.S. Census Bureau; U.S. BLS for CPI/PPI) is a two-stage procedure. First, a **`regARIMA`** model removes deterministic calendar and outlier effects and extends the series with forecasts/backcasts:

​$\phi(L)\,\Phi(L^{12})\,(1-L)^{d}(1-L^{12})^{D}\!\left(y_t - \textstyle\sum_i \beta_i x_{i,t}\right) = \theta(L)\,\Theta(L^{12})\,a_t ,$​

where the regressors $x_{i,t}$ capture **trading-day** and **length-of-month/leap-year** effects, **moving holidays** (notably Easter — material for HICP), and **outliers**: additive outliers (AO), level shifts (LS), temporary changes (TC), and ramps. Second, either the iterated X-11 moving-average filters or the model-based **SEATS** signal-extraction step (§8.4) produces the seasonal component. The canonical parsimonious specification is the **airline model** $(0,1,1)(0,1,1)_{12}$:

​$(1-L)(1-L^{12})\,y_t \;=\; (1-\theta L)(1-\Theta L^{12})\,a_t .$​

Outlier handling is central to §10. _Advantages_: rigorous, established, jointly handles calendar effects and outliers, permits slowly evolving seasonality. _Limitations_: engineered for statistical production rather than derivative pricing — the output must be converted into a forward-applicable, (3.1)-normalised factor — and the _concurrent_ seasonal estimate is revised as data accrue.

### 8.4 TRAMO-SEATS (ARIMA-model-based signal extraction)

TRAMO ("Time-series Regression with ARIMA noise, Missing observations and Outliers") performs the pre-adjustment analogue of `regARIMA`; SEATS ("Signal Extraction in ARIMA Time Series") then derives the Wiener–Kolmogorov optimal estimators of the unobserved components from the _partial-fraction decomposition of the fitted ARIMA spectrum_ (Hillmer & Tiao, 1982; Gómez & Maravall, 1996). The reference platform is JDemetra+ (the Eurostat-recommended tool). Across 36 OECD economies, differences between X-13 and TRAMO-SEATS under default specifications are small in normal times, as are the differences between the _direct_ (adjust the aggregate) and _indirect_ (aggregate the adjusted components) approaches (Arend et al., 2024).

### 8.5 STL (Seasonal-Trend decomposition using Loess)

STL (Cleveland et al., 1990) decomposes $y_t=\tau_t+s_t+\iota_t$ by iterated locally-weighted regression (Loess), with an inner loop that smooths the cycle-subseries (seasonal) and the deseasonalised series (trend) alternately, and an outer loop of robustness weights that downweight outliers. Explicit smoothing windows (seasonal, trend, low-pass) allow seasonality to evolve at a controlled rate and confer robustness — useful for exploratory work and where the pattern is known to drift.

### 8.6 Structural (state-space) time-series models

The **Basic Structural Model** (BSM; Harvey, 1989; Durbin & Koopman, 2012) treats trend and seasonal as _stochastic unobserved components_ and is, given a quant desk's existing Kalman-filter infrastructure, a natural framework for evolving seasonality with explicit uncertainty. A local-linear-trend plus trigonometric-stochastic-seasonal specification is

​$\begin{aligned} y_t &= \tau_t + s_t + \varepsilon_t, && \varepsilon_t \sim \mathrm{N}(0,\sigma^2_\varepsilon), \\[2pt] \tau_t &= \tau_{t-1} + \beta_{t-1} + \eta_t, && \eta_t \sim \mathrm{N}(0,\sigma^2_\eta), \\ \beta_t &= \beta_{t-1} + \zeta_t, && \zeta_t \sim \mathrm{N}(0,\sigma^2_\zeta), \\[2pt] s_t &= \textstyle\sum_{j=1}^{6} s_{j,t}, && \begin{pmatrix} s_{j,t} \\ s_{j,t}^{*} \end{pmatrix} = \begin{pmatrix} \cos\lambda_j & \sin\lambda_j \\ -\sin\lambda_j & \cos\lambda_j \end{pmatrix} \!\begin{pmatrix} s_{j,t-1} \\ s_{j,t-1}^{*} \end{pmatrix} + \begin{pmatrix} \omega_{j,t} \\ \omega_{j,t}^{*} \end{pmatrix}, \end{aligned}$​

with $\lambda_j = 2\pi j/12$ and $\omega_{j,t},\omega^*_{j,t}\sim \mathrm{N}(0,\sigma^2_\omega)$. An equivalent **dummy** form writes $U(L)\,s_t=\omega_t$, i.e. $s_t=-\sum_{j=1}^{11}s_{t-j}+\omega_t$, which enforces (3.1) _in expectation_ at every date. The hyperparameters $(\sigma^2_\eta,\sigma^2_\zeta,\sigma^2_\omega,\sigma^2_\varepsilon)$ are estimated by maximum likelihood via the Kalman filter (recursions in Appendix B), and the smoothed seasonal factor with its variance follows directly. **Deterministic seasonality is the nested restriction** $\sigma^2_\omega=0$, which is _testable_ (a boundary likelihood-ratio test), turning the §6 dichotomy into a parameter test. _Advantages_: coherent forecasts with uncertainty bands, principled control of the evolution rate via the seasonal signal-to-noise ratio $\sigma^2_\omega/\sigma^2_\varepsilon$, and native enforcement of the normalisation. _Limitations_: more demanding estimation and the need to embed the filtered/forecast component correctly in the curve build.

### 8.7 SARIMA / airline benchmark

A pure forecasting alternative is the seasonal ARIMA $(p,d,q)(P,D,Q)_{12}$, of which the airline model (8.3) is the canonical special case. It captures stochastic seasonality through the $(1-L^{12})$ factor and is the standard benchmark against which more elaborate models (including machine learning) are validated.

### 8.8 Bottom-up component aggregation

Model the seasonal factor of each major COICOP/CPI subcomponent — energy, food, core goods, apparel, transport services (airfares), accommodation, shelter — separately, then aggregate by expenditure weights $w_{c,t}$:

​$\hat s_t \;=\; \log\!\Big(\textstyle\sum_c w_{c,t}\, e^{\,\hat s_{c,t}}\Big) \quad\text{(or the additive approximation } \textstyle\sum_c w_{c,t}\,\hat s_{c,t}).$​

Energy, apparel, and travel components carry the bulk of the headline seasonal signal; component-level modelling is more robust to _annual re-weighting_ (the weights $w_{c,t}$ update each year) and diagnoses _which_ component drives a print surprise, at the cost of additional data infrastructure and the direct-vs-indirect consistency question flagged in §8.4.

### 8.9 Machine-learning approaches

A recent strand fits deep sequence models — notably **LSTM networks** — to the seasonal component of inflation-indexed swaps, training so as to respect the statistical and econometric structure (including the normalisation), benchmarking against a SARIMA robustness model, and using the two methodologies to bracket the fair value of a YYIIS and thereby _quantify model risk_ (see the DOAJ/ResearchGate working paper _"Deep Learning for seasonality modelling in Inflation-Indexed Swap pricing,"_ 2021). Hybrid SARIMA–LSTM–gradient-boosting stacks have since appeared in the CPI-forecasting literature. These methods can capture nonlinear and evolving structure but at the cost of opacity, overfitting risk, and materially harder enforcement of the no-arbitrage normalisation (3.1); they are best treated as a complement to, not a replacement for, the structural approaches above.

***

## 9. Curve construction and pricing consistency

The seasonal model is only useful if it is integrated with the forward-curve bootstrap _coherently_. The standard pipeline:

1. **Strip.** Deseasonalise the historical NSA index using the _same_ decomposition (§8.3–8.6) that will later re-apply the factor, obtaining a smooth trend proxy.
2. **Bootstrap the deseasonalised forward.** From the liquid ZCIIS quotes at annual pillars, recover the deseasonalised forward index $\hat\tau_{T_M}$ via (4.1); interpolate $\hat\tau_t$ between pillars (log-linear on the index, or a smoother spline, per desk convention).
3. **Re-impose seasonality.** Form $\mathcal{F}_I(0,t)=\exp(\hat\tau_t+\hat s_t)$ as in (4.2).
4. **Verify exact repricing.** By (3.1) the factor integrates out at the annual pillars; confirm that every calibrating ZCIIS reprices to zero to numerical tolerance. Any residual signals a normalisation violation or an inconsistency between the strip and re-apply steps.

Consistency between the _stripping_ method and the _re-application_ is central: using, say, X-13 factors to strip but a bespoke dummy vector to re-apply introduces a hidden basis. For ILBs, the daily reference index (4.4) then follows mechanically from the monthly $\mathcal{F}_I$ values at the lagged reference months, with no additional intramonth seasonal overlay.

***

## 10. Estimation pitfalls and the pandemic structural break

**The COVID-19 window (2020–2022) is the primary estimation hazard in current calibrations.** Lockdowns, reopening, and the subsequent energy shock injected large additive outliers, level shifts, and ramps that — if left untreated — contaminate _every_ calendar month's estimated factor, because the seasonal filters attribute part of the shock to the seasonal pattern. The agency response is **intervention analysis**: within X-13ARIMA-SEATS one pre-specifies AO/LS/ramp regressors so that the statistical significance and magnitude of non-seasonal events are estimated and their effects removed _before_ seasonal factors are computed, with the seasonal factors then applied to the original (intervention-inclusive) series (U.S. BLS, 2022, 2024). The OECD review documents that this outlier treatment _materially changes_ the resulting factors and recommends that the specifications be revisited once the shock's effect on inflation has fully dissipated (Arend et al., 2024). The analyst therefore faces an explicit choice among: (i) include the window with outlier correction; (ii) exclude the affected span; or (iii) downweight it (robust estimation / STL robustness weights).

Other recurring pitfalls:

* **Sample length versus relevance.** Eleven free factors demand many years, but structural change (§7) argues for recency. State-space evolution (§8.6) or Bayesian shrinkage across months resolves the tension without an arbitrary cut-off.
* **Annual re-weighting and methodology changes.** CPI basket weights update on a fixed cadence; HICP re-weights annually; treat each re-weighting as a potential break in the _component_ seasonal mix (favouring §8.8).
* **Calendar effects beyond month-of-year.** Trading/working-day, moving-holiday (Easter for HICP), and leap-year effects must be modelled (as in (8.2)); a naive twelve-vector folds shifting Easter into March/April and biases those factors.
* **Index-specific conventions.** U.S. CPI-U NSA is _not revised_ post-publication (a useful property for reproducible marks); HICPxT is revisable; U.K. RPI carries the well-known "formula effect" and, critically, a **scheduled structural break** — the RPI reform aligning its methodology to CPIH from **February 2030** — which should already be provisioned in long-dated RPI books.
* **Revisions imported from agency factors.** Relying on published SA factors imports their annual revisions (concurrent vs. final); estimating in-house avoids this but transfers full model-risk ownership to the desk.

***

## 11. Market-implied seasonality and relative value

The operationally relevant question is not "what is the statistically estimated seasonal factor?" but "how does the model estimate differ from the seasonality the market is pricing?" One inverts a strip of short-dated ZCIIS and monthly-fixing quotes (and dedicated seasonal swaps) to extract the **market-implied** seasonal vector $\{S^{\text{impl}}_m\}$, and compares it to the model vector:

​$\Delta_m \;=\; \hat S^{\text{model}}_m - S^{\text{impl}}_m , \qquad \text{signal} \iff |\Delta_m| \;>\; \tfrac12\,(\text{bid–offer}) + \text{estimation s.e.}$​

The gap $\Delta_m$ is simultaneously the relative-value signal _and_ the risk; because desks tend to converge on similar vectors (often anchored to a common vendor), idiosyncratic seasonal views are a genuine but crowded source of front-end alpha. The estimation standard error entering (11.1) is available directly from the state-space model (§8.6) and by bootstrap for the deterministic estimators.

***

## 12. Model-risk quantification

Since the seasonal choice moves front-end marks, the seasonal method itself is a model-risk axis and should be reserved against. Reprice the short-dated book under a plausible set of methods $\mathcal{M}=\{\text{dummy vector},\ \text{X-13},\ \text{TRAMO-SEATS},\ \text{state-space},\ \text{market-implied}\}$ and carry the dispersion as a valuation adjustment:

​$\mathrm{VA}_{\text{seas}} \;=\; \sup_{m\in\mathcal{M}}\big|\,V\!\big(\text{book};\,\hat s^{(m)}\big) - V\!\big(\text{book};\,\hat s^{(\text{base})}\big)\big|,$​

or, less conservatively, an inter-quartile spread across $\mathcal{M}$. This is the exercise the machine-learning IIS studies perform for a single YYIIS (§8.9), generalised to the book.

***

## 13. Recommended workflow

1. **Data.** Obtain the long NSA history for the _contractually referenced_ index (CPI-U, HICPxT, or RPI).
2. **Pre-adjust.** Fit `regARIMA` (8.2) for trading-day, Easter, and leap-year effects, with intervention outliers for 2020–2022 (§10).
3. **Detect and classify.** Presence: $F$-dummy (5.1), Kruskal–Wallis (5.2), $QS$ (5.3), spectrum (5.4), $M7$ (5.5). Character: HEGY (6.2) and Canova–Hansen (6.3) jointly, plus OCSB; stability via moving-seasonality $F$, sliding spans, and Bai–Perron (§7).
4. **Estimate.** X-13 or TRAMO-SEATS for a production factor; a structural model (8.4) where evolution or explicit uncertainty is required; test $\sigma^2_\omega=0$ to confirm the deterministic/stochastic verdict.
5. **Reduce and normalise.** Collapse to a smoothed or harmonic factor and enforce the rolling-twelve constraint (3.1).
6. **Integrate.** Bootstrap the deseasonalised forward from ZCIIS quotes, re-impose the factor (4.2), and verify exact repricing of the calibration set (§9).
7. **Benchmark.** Extract market-implied seasonality and size the divergence $\Delta_m$ against bid–offer plus estimation error (11.1).
8. **Reserve.** Quantify $\mathrm{VA}_{\text{seas}}$ across methods (12.1).
9. **Monitor.** Re-estimate on a fixed cadence; flag re-weightings, methodology changes, and the RPI→CPIH break scheduled for February 2030.

***

## Appendix A. The quarterly HEGY regression (explicit form)

For quarterly data $1-L^4=(1-L)(1+L)(1+L^2)$, with frequencies $0$, $\pi$ (semi-annual), and $\pi/2$ (annual, a complex pair). Define the frequency-isolating filters

​$\begin{aligned} y_{1,t} &= (1+L)(1+L^2)\,y_t = (1+L+L^2+L^3)\,y_t, \\ y_{2,t} &= -(1-L)(1+L^2)\,y_t = -(1-L+L^2-L^3)\,y_t, \\ y_{3,t} &= -(1-L)(1+L)\,y_t = -(1-L^2)\,y_t, \\ y_{4,t} &= (1-L^4)\,y_t . \end{aligned}$​

The HEGY (1990) auxiliary regression is

​$y_{4,t} \;=\; \pi_1\,y_{1,t-1} + \pi_2\,y_{2,t-1} + \pi_3\,y_{3,t-2} + \pi_4\,y_{3,t-1} + \sum_{j}\varphi_j\,y_{4,t-j} + \mu_t + e_t ,$​

with $\mu_t$ the chosen deterministics. Tests: $\pi_1=0$ (zero-frequency unit root, one-sided $t$); $\pi_2=0$ (root at $\pi$, one-sided $t$); $\pi_3=\pi_4=0$ (complex pair at $\pi/2$, joint $F$). Non-rejection of a $\pi$ indicates the corresponding seasonal unit root is present. Critical values are tabulated in HEGY (1990) and depend on $\mu_t$ and sample size; the monthly generalisation (Beaulieu & Miron, 1993) applies the same logic to the twelve frequencies of (6.1).

## Appendix B. State-space form and Kalman recursions for the BSM

Stack the states of (8.4) into $\alpha_t=(\tau_t,\beta_t,s_{1,t},s_{1,t}^*,\dots,s_{6,t})'$ with system

​$\alpha_t = T\,\alpha_{t-1} + R\,\xi_t,\qquad y_t = Z\,\alpha_t + \varepsilon_t,\qquad \xi_t\sim\mathrm{N}(0,Q),\ \varepsilon_t\sim\mathrm{N}(0,\sigma^2_\varepsilon),$​

where $T$ is block-diagonal (a $2\times2$ trend block $\left[\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right]$ and the six $2\times2$ rotation blocks of (8.4)), $Z$ selects $\tau_t$ and each $s_{j,t}$, and $Q=\mathrm{diag}(\sigma^2_\eta,\sigma^2_\zeta,\sigma^2_\omega,\dots)$. The Kalman filter (prediction and update, the latter in numerically-stable Joseph form) is

​$\begin{aligned} &\text{Predict:} && a_{t|t-1} = T\,a_{t-1}, && P_{t|t-1} = T\,P_{t-1}\,T' + R\,Q\,R', \\ &\text{Innovation:} && v_t = y_t - Z\,a_{t|t-1}, && F_t = Z\,P_{t|t-1}\,Z' + \sigma^2_\varepsilon, \\ &\text{Gain:} && K_t = P_{t|t-1}\,Z'\,F_t^{-1}, && \\ &\text{Update:} && a_t = a_{t|t-1} + K_t\,v_t, && P_t = (I - K_t Z)\,P_{t|t-1}(I - K_t Z)' + K_t\,\sigma^2_\varepsilon\,K_t' . \end{aligned}$​

Hyperparameters are estimated by maximising the prediction-error decomposition of the log-likelihood,

​$\log \mathcal{L} \;=\; -\frac{1}{2}\sum_{t=1}^{T}\Big[\log 2\pi + \log F_t + \frac{v_t^{2}}{F_t}\Big],$​

after which the Kalman smoother returns the seasonal component $\{\hat s_t\}$ and its variance. The restriction $\sigma^2_\omega=0$ yields deterministic seasonality and is tested by a boundary likelihood-ratio statistic.

***

## References

Arend, T., et al. (2024). _Seasonal Adjustment of CPIs during the COVID-19 Pandemic and Beyond._ OECD Statistics Working Papers, No. 2024/04. OECD Publishing, Paris. [https://doi.org/10.1787/dac02d60-en](https://doi.org/10.1787/dac02d60-en)​

Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. _Econometrica_, 66(1), 47–78.

Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. _Journal of Applied Econometrics_, 18(1), 1–22.

Beaulieu, J. J., & Miron, J. A. (1993). Seasonal unit roots in aggregate U.S. data. _Journal of Econometrics_, 55(1–2), 305–328.

Belgrade, N., & Benhamou, E. (2004). _Impact of Seasonality in Inflation Derivatives Pricing._ Working paper (CDC IXIS Capital Markets / SSRN).

Belgrade, N., Benhamou, E., & Koehler, E. (2004). _A Market Model for Inflation._ Working paper, SSRN.

Box, G. E. P., & Jenkins, G. M. (1970). _Time Series Analysis: Forecasting and Control._ Holden-Day, San Francisco.

Brigo, D., & Mercurio, F. (2006). _Interest Rate Models — Theory and Practice: With Smile, Inflation and Credit_ (2nd ed.), Ch. on inflation-indexed derivatives. Springer, Berlin.

Canova, F., & Hansen, B. E. (1995). Are seasonal patterns constant over time? A test for seasonal stability. _Journal of Business & Economic Statistics_, 13(3), 237–252.

Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on Loess. _Journal of Official Statistics_, 6(1), 3–73.

Deacon, M., Derry, A., & Mirfendereski, D. (2004). _Inflation-Indexed Securities: Bonds, Swaps and Other Derivatives_ (2nd ed.). Wiley, Chichester.

_Deep Learning for Seasonality Modelling in Inflation-Indexed Swap Pricing_ (2021). Working paper, available via DOAJ and ResearchGate. \[LSTM approach to IIS seasonality with SARIMA robustness benchmark and YYIIS model-risk quantification.]

Dickey, D. A., Hasza, D. P., & Fuller, W. A. (1984). Testing for unit roots in seasonal time series. _Journal of the American Statistical Association_, 79(386), 355–367.

Durbin, J., & Koopman, S. J. (2012). _Time Series Analysis by State Space Methods_ (2nd ed.). Oxford University Press, Oxford.

Findley, D. F., Monsell, B. C., Bell, W. R., Otto, M. C., & Chen, B.-C. (1998). New capabilities and methods of the X-12-ARIMA seasonal-adjustment program. _Journal of Business & Economic Statistics_, 16(2), 127–152.

Franses, P. H. (1991). Seasonality, non-stationarity and the forecasting of monthly time series. _International Journal of Forecasting_, 7(2), 199–208.

Franses, P. H. (1996). _Periodicity and Stochastic Trends in Economic Time Series._ Oxford University Press, Oxford.

Gómez, V., & Maravall, A. (1996). _Programs TRAMO and SEATS: Instructions for the User._ Working Paper 9628, Banco de España.

Harvey, A. C. (1989). _Forecasting, Structural Time Series Models and the Kalman Filter._ Cambridge University Press, Cambridge.

Hillmer, S. C., & Tiao, G. C. (1982). An ARIMA-model-based approach to seasonal adjustment. _Journal of the American Statistical Association_, 77(377), 63–70.

Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990). Seasonal integration and cointegration. _Journal of Econometrics_, 44(1–2), 215–238.

Jarrow, R., & Yildirim, Y. (2003). Pricing Treasury inflation protected securities and related derivatives using an HJM model. _Journal of Financial and Quantitative Analysis_, 38(2), 337–358.

Kerkhof, J. (2005). _Inflation Derivatives Explained: Markets, Products, and Pricing._ Fixed Income Quantitative Research, Lehman Brothers.

Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. _Journal of Econometrics_, 54(1–3), 159–178.

Ladiray, D., & Quenneville, B. (2001). _Seasonal Adjustment with the X-11 Method._ Lecture Notes in Statistics 158. Springer, New York.

Mercurio, F. (2005). Pricing inflation-indexed derivatives. _Quantitative Finance_, 5(3), 289–302.

Mercurio, F., & Moreni, N. (2006). Inflation with a smile. _Risk_, 19(3), 70–75.

Osborn, D. R., Chui, A. P. L., Smith, J. P., & Birchenhall, C. R. (1988). Seasonality and the order of integration for consumption. _Oxford Bulletin of Economics and Statistics_, 50(4), 361–377.

Parameta Solutions (2025). _Inflation Derivatives — Inflation Swaps Data._ Product documentation. [https://www.parametasolutions.com/solutions/capital-markets/inflation-derivatives/](https://www.parametasolutions.com/solutions/capital-markets/inflation-derivatives/)​

U.S. Bureau of Labor Statistics (2022). PPI and CPI seasonal adjustment during the COVID-19 pandemic. _Monthly Labor Review_, May 2022.

U.S. Bureau of Labor Statistics (2024). _Intervention Analysis in Seasonal Adjustment._ CPI Seasonal Adjustment documentation.

U.S. Census Bureau. _X-13ARIMA-SEATS Reference Manual._ U.S. Department of Commerce.
