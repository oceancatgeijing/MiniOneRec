# Final Experiment Summary

## Scope

This summary covers the completed MiniOneRec experiments on the remote RTX PRO 6000 environment:

1. Office official checkpoint evaluation
2. Industrial official checkpoint evaluation
3. Office SFT full_1epoch
4. Office SFT continue_epoch2
5. Office GRPO debug

Metric order is always `@1/@3/@5/@10/@20/@50`.

## Experiment Overview

| Experiment | Dataset | Train rows | Eval rows | Test rows | Train steps | Train loss | Eval loss | Peak GPU memory | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Office official checkpoint | Office_Products | N/A | N/A | 4866 | N/A | N/A | N/A | not recorded | eval runtime not recorded |
| Industrial official checkpoint | Industrial_and_Scientific | N/A | N/A | 4533 | N/A | N/A | N/A | not recorded | eval runtime not recorded |
| SFT full_1epoch | Office_Products | 84735 | 4866 | 4866 | 1324 | 0.9007771231931863 | 1.4612122774124146 | about 20.3 GiB | 1912.106s |
| SFT continue_epoch2 | Office_Products | 84735 | 4866 | 4866 | 1324 | 0.5008011465706494 | 1.380003571510315 | about 20.3 GiB | 1936.9656s |
| GRPO debug | Office_Products | 384 | 128 | 4866 | 384 | 0.00021698789579479202 | 0.002977680414915085 | about 45.1 GiB | 188.2877s |

## Recommendation Metrics

| Experiment | Dataset | HR@1/3/5/10/20/50 | NDCG@1/3/5/10/20/50 | CC |
|---|---|---|---|---:|
| Office official checkpoint | Office_Products | [0.0945335, 0.12597616, 0.13892314, 0.15885738, 0.18392931, 0.23715577] | [0.0945335, 0.112811, 0.11826087, 0.12468961, 0.13097781, 0.14141109] | 0 |
| Industrial official checkpoint | Industrial_and_Scientific | [0.08515332, 0.11294948, 0.13236267, 0.15729098, 0.19258769, 0.24442974] | [0.08515332, 0.10101549, 0.10905726, 0.11699515, 0.12591518, 0.13612897] | 0 |
| SFT full_1epoch | Office_Products | [0.06699548, 0.10213728, 0.11816687, 0.14673243, 0.18228524, 0.24167694] | [0.06699548, 0.08776832, 0.09432966, 0.10353097, 0.11246303, 0.12418309] | 0 |
| SFT continue_epoch2 | Office_Products | [0.07727086, 0.10686395, 0.12227702, 0.13851212, 0.16420058, 0.21701603] | [0.07727086, 0.09491955, 0.10129643, 0.10650435, 0.1128923, 0.12339573] | 0 |
| GRPO debug | Office_Products | [0.04788327, 0.08343609, 0.0998767, 0.12309905, 0.15269215, 0.20838471] | [0.04788327, 0.06883471, 0.0756181, 0.08309315, 0.0904672, 0.10146564] | 0 |

## Notes

- Official checkpoint rows refer to test samples used by `evaluate.py` and `calc.py`; these checkpoints were not trained in this run, so training rows, train loss, eval loss, GPU memory, and training runtime are not applicable.
- SFT `eval rows` are validation rows used during training; recommendation metrics are computed on the Office test set with 4866 samples.
- GRPO debug used a reduced debug dataset: 384 train rows and 128 eval rows, with `num_generations=4`, `reward_type=ranking`, and `beam_search=True`.
- GRPO debug evaluation was still run on the full Office test set with 4866 samples.

## Conclusion

1. SFT epoch1 is the best Office self-trained checkpoint among the completed self-training runs when judged by HR@50. It reached HR@50 = 0.24167694, slightly above the Office official checkpoint reproduction HR@50 = 0.23715577.
2. continue_epoch2 reduced SFT loss from 0.9008 to 0.5008 and eval_loss from 1.4612 to 1.3800, but HR@10/20/50 dropped. This confirms that lower SFT loss does not necessarily translate to better HR/NDCG across the ranking depth.
3. GRPO debug successfully completed the SFT -> GRPO/RL chain on a single RTX PRO 6000, including beam search, num_generations=4, ranking reward, rule reward, KL constraint, checkpoint saving, and later evaluation. Its metrics are lower because it was only a small debug run, not a final RL experiment.
4. The best Office HR@50 observed in these experiments is 0.2417 from SFT full_1epoch.
