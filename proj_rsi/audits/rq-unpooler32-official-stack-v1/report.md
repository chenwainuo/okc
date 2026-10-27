# RQ unpooler32: actual combined CHECK supplement

CHECK71a76a2c3fcd431f passes9/9 at3182.348773us. It is25.310370us slower than scored incumbent9916d34357b745cf and43.544472us faster than keyed-half CHECKac49231d41d441f6. This is a valid mixed transfer result, not an incumbent win.

The new guarded rq<32,6,1,0,0,0,131107> actually executes. Bake selected19.20us against original ST6 19.33us and ST5 19.39us; prepare finished163s. Thus this attempt is not a source-inclusion or budget-exposure failure. Prior local paired+0.048us/F1+0.080us is retained.

Positive gain below favors the new package; all cross-job deltas are descriptive.

| Control | Target raw gain us | Target calibrated gain us | Changed seeds | Visible route changes |
|---|---:|---:|---:|---:|
| 9916d34357b745cf | -0.064000 | -0.015615 | 30/30 | 14/30 |
| ac49231d41d441f6 | +0.256000 | +0.607871 | 30/30 | 11/30 |

Against9916,6/30 raw and12/30 calibrated cases improve; largest load-sum losses are verify32−2.160876us, decode1−1.158078us and decode32−0.713025us. Againstac492,14/30 raw and20/30 calibrated cases improve. Full30 rows and both route lists are saved in CSV/analysis.json. The controls choose different RQ types:9916 ST6, ac492 ST5.

Calibrated time is calibrated baseline divided by paired ratio, not raw median minus fixed overhead. At target, new raw19.072us/calibrated15.074713us/overhead4.000us;9916 raw19.008us/calibrated15.059097us/overhead3.936us;ac492 raw19.328us/calibrated15.682584us/overhead3.584us. All30 input seeds change. Raw, calibrated, bake and profile times must remain separate.

PROVEN: valid official outputs; new variant selected; cross-job seeds/calibration/routes differ. UNKNOWN: unique implementation/scheduling contribution or cause of official transfer. Earlier native equality and numerical checks do not prove scheduling or package transfer. The smallest later falsifier is one same-run fixed-unpooler contrast of actual incumbentST6, ac492-selectedST5 and new131107 with shared fixture and current-input/own-output checks; no unchanged full30 retry is proposed.

The separate compact-half resource proposal targets a real256-thread kernel with216 math/40 producer budgets, rather than the historical384-thread HALF early exit. Arithmetic permits two32768-register/62624-shared-byte CTAs, but actual compile/lifetime/occupancy are UNKNOWN. Two compact CTAs retain only two active math WGs, while doubling X traffic and producer WGs. Old HALF losses and handshake serialization are explicit counterevidence. First step is CPU role proof and one offline compile, not GPU admission.

No GPU/API/SSH/queue/git action. Earlier frozen evidence is unchanged.

Loop: profile → measure → hypothesis → optimize/test → validate → stack/submit → learn → reprioritize → repeat. [Contract](../../ITERATION.md).
