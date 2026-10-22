# Reviewed case catalog

These are 20 case-specific tasks, not 20 universal family-wide cutoffs. The linked JSON is authoritative for target state, atom roles, numerical bounds, literature support, author-defined tolerances and exclusions.

Protein copy count does not by itself specify ligand occupancy. Read the input contract; for example, a one-pair task in a multimer is not a requirement that all potential sites are occupied.

| Case | M-CSA | Net-reaction family | Protein copies | Definition | Evaluator |
| --- | --- | --- | ---: | --- | --- |
| 1ZIO | [M0290](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/290/) | `TR_PHOSPHORYL` | 1 | [JSON](../src/mcsa_benchmark/data/1ZIO.json) | [source](../src/mcsa_benchmark/mechanisms/adenylate.py) |
| 1NDP | [M0150](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/150/) | `TR_PHOSPHORYL` | 6 | [JSON](../src/mcsa_benchmark/data/1NDP.json) | [source](../src/mcsa_benchmark/mechanisms/ndpk_histidine.py) |
| 1OTH | [M0012](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/12/) | `TR_ACYL` | 3 | [JSON](../src/mcsa_benchmark/data/1OTH.json) | [source](../src/mcsa_benchmark/mechanisms/ornithine_carbamoyl.py) |
| 1B6B | [M0022](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/22/) | `TR_ACYL` | 1 | [JSON](../src/mcsa_benchmark/data/1B6B.json) | [source](../src/mcsa_benchmark/mechanisms/aanat_acyl.py) |
| 1XVA | [M0023](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/23/) | `TR_SAM_METHYL` | 4 | [JSON](../src/mcsa_benchmark/data/1XVA.json) | [source](../src/mcsa_benchmark/mechanisms/sam_n_methyl.py) |
| 1GSA | [M0199](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/199/) | `AD_NTP_CARBOXYLATE_LIGATION` | 2 | [JSON](../src/mcsa_benchmark/data/1GSA.json) | [source](../src/mcsa_benchmark/mechanisms/glutathione_synthesis.py) |
| 1KQP | [M0200](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/200/) | `AD_NTP_CARBOXYLATE_LIGATION` | 2 | [JSON](../src/mcsa_benchmark/data/1KQP.json) | [source](../src/mcsa_benchmark/mechanisms/nad_synthetase.py) |
| 1DHP | [M0267](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/267/) | `AD_CC` | 4 | [JSON](../src/mcsa_benchmark/data/1DHP.json) | [source](../src/mcsa_benchmark/mechanisms/lys_carbonyl_multisite.py) |
| 1IGS | [M0252](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/252/) | `AD_CYCLIZATION` | 1 | [JSON](../src/mcsa_benchmark/data/1IGS.json) | [source](../src/mcsa_benchmark/mechanisms/igps_nac.py) |
| 5ICD | [M0007](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/7/) | `RX_GENERAL` | 2 | [JSON](../src/mcsa_benchmark/data/5ICD.json) | [source](../src/mcsa_benchmark/mechanisms/isocitrate.py) |
| 1LDM | [M0584](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/584/) | `RX_GENERAL` | 4 | [JSON](../src/mcsa_benchmark/data/1LDM.json) | [source](../src/mcsa_benchmark/mechanisms/ldh_tetramer.py) |
| 1IDT | [M0211](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/211/) | `RX_GENERAL` | 2 | [JSON](../src/mcsa_benchmark/data/1IDT.json) | [source](../src/mcsa_benchmark/mechanisms/nitroreductase_hydride.py) |
| 1XEL | [M0188](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/188/) | `IS_STEREO` | 1 | [JSON](../src/mcsa_benchmark/data/1XEL.json) | [source](../src/mcsa_benchmark/mechanisms/gale_hydride.py) |
| 1I8T | [M0368](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/368/) | `IS_RING_REORGANIZATION` | 2 | [JSON](../src/mcsa_benchmark/data/1I8T.json) | [source](../src/mcsa_benchmark/mechanisms/ugm_n5.py) |
| 1QFE | [M0054](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/54/) | `CL_ELIMINATION` | 2 | [JSON](../src/mcsa_benchmark/data/1QFE.json) | [source](../src/mcsa_benchmark/mechanisms/dhqd_lys_carbonyl.py) |
| 1ET0 | [M0305](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/305/) | `CL_ELIMINATION` | 2 | [JSON](../src/mcsa_benchmark/data/1ET0.json) | [source](../src/mcsa_benchmark/mechanisms/plp_transaldimination.py) |
| 1QQ5 | [M0036](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/36/) | `HY_OTHER` | 2 | [JSON](../src/mcsa_benchmark/data/1QQ5.json) | [source](../src/mcsa_benchmark/mechanisms/haloacid_sn2.py) |
| 1L7N | [M0727](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/727/) | `HY_PHOSPHATE` | 1 | [JSON](../src/mcsa_benchmark/data/1L7N.json) | [source](../src/mcsa_benchmark/mechanisms/phosphoserine_phosphatase.py) |
| 1IM5 | [M0716](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/716/) | `HY_ACYL` | 1 | [JSON](../src/mcsa_benchmark/data/1IM5.json) | [source](../src/mcsa_benchmark/mechanisms/nicotinamidase.py) |
| 2GSA | [M0195](https://www.ebi.ac.uk/thornton-srv/m-csa/entry/195/) | `IS_OTHER` | 2 | [JSON](../src/mcsa_benchmark/data/2GSA.json) | [source](../src/mcsa_benchmark/mechanisms/pmp_carbonyl.py) |

## Frozen configuration

`criteria_lock.json` records the reviewed configuration hashes. The test suite verifies each primary configuration and the shared general policy against those hashes. Changing scientific content requires an explicitly documented criterion revision, not a silent lock update.

The legacy private primary-manifest hash is provenance, not a promise that the coordinate archive is distributed in this repository. See [data policy](data_policy.md).
