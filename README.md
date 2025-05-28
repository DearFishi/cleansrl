# Safety-RL

## Benchmarks

<table>
    <tr>
        <th rowspan="2">Env</th>
        <th colspan="2">PPO</th>
        <th colspan="2">PPO_Lagrangian</th>
        <th colspan="2">PPO_CRPO</th>
        <th colspan="2">PPO_CRPO_SR</th>
    </tr>
    <tr>
        <td>Return</td>
        <td>Cost</td>
        <td>Return</td>
        <td>Cost</td>
        <td>Return</td>
        <td>Cost</td>
        <td>Return</td>
        <td>Cost</td>
    </tr>
    <tr>
        <td>SafetyHalfCheetahVelocity-v1</td>
        <td>4927.870±1265.715</td>
        <td>823.125±219.088</td>
        <td>2074.117±748.714</td>
        <td>123.192±69.233</td>
        <td>608.081±425.514</td>
        <td>14.582±17.990</td>
        <td>611.925±259.232</td>
        <td>0.032±0.203</td>
    </tr>
    <tr>
        <td>SafetySwimmerVelocity-v1</td>
        <td>43.901±18.263</td>
        <td>36.776±29.344</td>
        <td>-4.965±5.780</td>
        <td>0.321±2.787</td>
        <td>15.633±3.435</td>
        <td>34.992±14.742</td>
        <td>17.970±6.626</td>
        <td>2.469±5.474</td>
    </tr>
    <tr>
        <td>SafetyPointGoal1-v0</td>
        <td>11.829±6.338</td>
        <td>95.632±112.685</td>
        <td>-9.743±8.966</td>
        <td>10.093±34.616</td>
        <td>23.726±3.107</td>
        <td>64.762±43.774</td>
        <td>7.047±4.763</td>
        <td>32.575±45.225</td>
    </tr>
</table>


## Supported algorithms
1. PPO :  [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
2. PPO_Largrangian : [Benchmarking Safe Exploration in Deep Reinforcement Learning ](https://cdn.openai.com/safexp-short.pdf)
3. PPO_CRPO : [CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee](https://arxiv.org/pdf/2011.05869)
4. PPO_CRPO_SR : [SAFETY REPRESENTATIONS FOR SAFER POLICY LEARNING](https://openreview.net/pdf?id=gJG4IPwg6l)