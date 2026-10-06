# UTWF simulation

This code runs the simulation study for the paper *Urban Traffic Waveform Fingerprinting (UTWF)*. Every number and figure in `results/` comes from running it. The data is synthetic, so the results describe the simulator, not any real city. Say this clearly in the paper.

## Run it

```
pip install -r requirements.txt
python utwf_sim.py                 # full study, ~1 minute
python utwf_sim.py --seeds 2 --quick   # fast check
```

The run is deterministic: the same seeds always give the same numbers. All parameters are saved to `results/config.json`.

## What it does

1. **Builds three cities** of 36 junctions each: grid, radial (spokes plus two ring roads) and organic (random geometric graph).
2. **Generates traffic** at 5-minute resolution. Each junction gets a local inbound demand that peaks in the morning and an outbound demand that peaks in the evening. Weekends get a flatter shape. Flow then passes between junctions through linear transfer functions: downstream flow equals a share `a_ij` of upstream flow, delayed by the link's travel time `tau_ij`. The data includes correlated demand noise, day-to-day variation and sensor noise.
3. **Fits the UTWF model** on 28 clean days:
   - *Junction signal model:* least-squares fit of 6 daily harmonics per junction, with separate weekday and weekend shapes.
   - *City waveform fingerprint:* normalised harmonic amplitudes of the aggregate weekday flow, plus a measure of how spread out junction peak phases are.
4. **Injects congestion** (20 events per week) into the following 21 days. Each event cuts flow at one junction by 15–60% for 30–180 minutes, with spillback onto neighbouring junctions.
5. **Detects congestion** from residuals against the fingerprint: robust z-scores, then a city-wide common-mode correction, temporal smoothing and neighbour (spatial) averaging. An alarm requires 15 minutes of persistence.
6. **Evaluates** against baselines. Thresholds are chosen on week 1 of the event period, and results are reported on the 2 weeks after that.

## Methods compared

| Method | What it is |
|---|---|
| UTWF | the full method |
| UTWF (no spatial) | ablation: no neighbour averaging |
| UTWF (no common-mode) | ablation: no city-wide demand correction |
| Historical average + CM | mean flow per time-of-week slot, with the same common-mode correction |
| Historical average | standard baseline |
| Rolling z-score | deviation from the trailing one-hour mean |

## Outputs

- `results/summary.md`: tables ready to paste into the paper
- `results/metrics_summary.csv`, `metrics_by_run.csv`, `model_fit.csv`, `fingerprint_classification.csv`
- `results/figures/`
  - fig1: the three city topologies
  - fig2: junction signal vs. harmonic fit over one week
  - fig3: city fingerprints and topology classification
  - fig4: one congestion event with the residual score
  - fig5: ROC curves

## Things to know before writing them up

- **Common-mode correction is new and it matters most.** Without it, low-demand days push every junction below its expected flow at once, and some seeds produce dozens of false alarms (F1 drops to about 0.73 on average, and below 0.4 in the worst seed). The paper's method section needs a paragraph on it. It is a natural fit with the fingerprint idea: congestion is a local deviation, while demand swings are global.
- **UTWF's F1 is level with historical average + CM.** Its advantage is ranking quality: ROC-AUC is about 0.98 vs. 0.96–0.97, and the ROC curves sit higher at low false-positive rates. It also uses far fewer parameters: 26 per junction, compared with 2,016 time-of-week slots per junction. Claim that, not "beats all baselines".
- **Spatial averaging helps clearly only on the organic topology.** On grid and radial it is neutral.
- **Fingerprint classification (80%) reflects the simulator's assumptions.** Present it as "the fingerprint can separate structurally different networks", not as a claim about real cities.
- **Not done yet:** the transfer coefficients `a_ij` and lags `tau_ij` are used to generate data but are not estimated back from it. Training data is assumed clean of congestion. Both are natural next steps or limitations to state.
