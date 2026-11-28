# Model Comparison

:API: {py:mod}`irtlib.comparison`
:Sources: {cite:t}`Akaike1974`; {cite:t}`Schwarz1978`; {cite:t}`Vrieze2012`; {cite:t}`Bock1981`

## Motivation

A model with more parameters always fits at least as well; the question is whether the improvement
outweighs the parameters spent. The likelihood-ratio test compares two nested models, such as the
2PL and the 3PL calibrated on the same responses; the information criteria rank any models
calibrated on them; and $G^2$ tests one model against every pattern of responses. BIC selects the
true model as the sample grows, when it is among the candidates, and AIC does not; AIC is efficient
when no candidate is true {cite:p}`Vrieze2012{Pages 10-11}`.

## Definition

For $k$ estimated parameters and the maximized log-likelihood $\log L$:

- the likelihood-ratio statistic of nested models is $-2$ times the difference of their
  log-likelihoods, "distributed as $\chi^2$" with the difference in their numbers of parameters as
  its degrees of freedom {cite:p}`Orlando2000{p. 51}`;
- AIC is $-2\log L + 2k$ {cite:p}`Akaike1974{p. 719}`;
- BIC is $-2\log L + k\log N$ for $N$ examinees, Schwarz's criterion
  {cite:p}`Schwarz1978{p. 461}` multiplied by $-2$, with the sample size
  {cite:t}`Vrieze2012{Page 10}` uses;
- $G^2$ compares the model with the general multinomial over response patterns,
  $2\sum_x r_x \log(r_x / (N \tilde P_x))$ over the observed patterns $x$, observed $r_x$ times with
  marginal probability $\tilde P_x$ {cite:p}`Bock1981{p. 450}`.

## Implementation

**$G^2$'s degrees of freedom** are $2^M - 1 - k$ for $M$ items, as in the values
{cite:t}`Bock1981{p. 455, Table 3}` and {cite:t}`Thissen1982{p. 178}` report, whatever the patterns
observed; Bock and Aitkin's Equation 16 (p. 450) counts the observed patterns instead. The
multinomial is not defined when responses are missing, so $G^2$ needs complete responses.

**The log-likelihoods.** With priors, the log-likelihood is the one at the Bayes modal estimates,
without the log-prior. Every statistic here takes marginal log-likelihoods, and refuses a calibration
of another likelihood, such as the conditional one of {doc}`/methods/calibration/cml`, which is not
on their scale ({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

:::{dropdown} Comparison with mirt

- mirt 1.48 counts $G^2$'s degrees of freedom as $2^M - 1 - k$ too.
- On calibrations of the same responses, the parity tests find the numbers of parameters equal to
  mirt's, the log-likelihoods, AIC and BIC within a relative $10^{-10}$, $G^2$ and the
  likelihood-ratio statistic within $10^{-6}$, and their degrees of freedom equal
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

The functions take calibrations; {func}`~irtlib.comparison.bic` also takes the number of examinees,
and {func}`~irtlib.comparison.g2` the responses. None has a setting.

## Guarantees

1. A calibration whose likelihood is not marginal raises an error.
2. The likelihood-ratio test raises an error unless the general model has more parameters than the
   restricted one, on the same items.
3. $G^2$ of responses with a missing entry raises an error.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.comparison
   :members:
```
