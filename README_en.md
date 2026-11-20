# Traffic signal control under changing demand

This experiment compares five ways to continue training a traffic signal controller under changing traffic demand. We use three controllers (MA2C, PPO and DQL) on two SUMO networks (a 5 × 5 grid and the Monaco subnet). An offline WCE policy learns to generate demand that increases queueing against a fixed signal controller.

**Review status, 20 November 2026.** Development tuning closed at the operator-selected 15:30 CET snapshot. Fourteen of the 24 candidates had reached 1,000 episodes at the cutoff. The other ten had shorter validation histories. The six selected configurations are ready for review. Formal training and the final test suite await manual release.

## Frozen training parameters

For each network and controller, we selected the candidate with the lowest recorded development validation mean queue at the cutoff. The score is the average number of stopped vehicles across monitored incoming lanes. The episode identifies the validation observation used for selection. Formal training starts from initialization with seed 101. Seed 9001 was used for development. The development policies and their scores are evidence for hyperparameter choice.

| Network | Controller | Candidate | Initial learning rate | Second parameter | Best mean queue | Episode |
|---|---|---:|---:|---|---:|---:|
| Grid | MA2C | 1 | 0.0005 | Initial entropy coefficient 0.001 | 26.483 | 1,000 |
| Grid | PPO | 2 | 0.00015 | Initial entropy coefficient 0.01 | 56.435 | 800 |
| Grid | DQL | 1 | 0.0001 | Replay capacity 20,000 | 138.001 | 700 |
| Monaco | MA2C | 1 | 0.0005 | Initial entropy coefficient 0.001 | 11.978 | 1,000 |
| Monaco | PPO | 2 | 0.00015 | Initial entropy coefficient 0.01 | 50.812 | 400 |
| Monaco | DQL | 0 | 0.0001 | Replay capacity 50,000 | 304.621 | 800 |

Candidates received unequal training budgets. These choices describe the recorded cutoff and may differ from the ranking after equal training budgets. The repository includes the full controller INIs and a selection manifest with configuration and validation checkpoint hashes. [Frozen configurations](config/revised/) · [Selection evidence](provenance/experiment02_selection.json)

## Training schedule

Each network–controller pair contributes one parent controller, one offline WCE and five continuations. This gives **42 training jobs** for one training seed. Each training episode spans 6,600 simulated seconds. The local experiment-02 scheduler admits up to 20 concurrent single-thread workers, subject to memory, storage and task dependencies.

| Stage | Jobs | Episodes per job | Starting policy and learning objective |
|---|---:|---:|---|
| Baseline parent | 6 | 5,000 | Train the signal controller and minimize its validation mean queue. |
| Offline WCE | 6 | 3,000 | Freeze the corresponding parent’s full-horizon best controller. Train the demand policy and maximize validation mean queue. |
| Baseline continuation | 6 | 5,000 additional | Resume the full-horizon best parent with the original cyclic demand schedule. |
| Random group | 6 | 5,000 additional | Resume the same parent. Sample a demand group per block. |
| Domain randomization | 6 | 5,000 additional | Resume the same parent. Sample mixture weights per block. |
| Fixed WCE | 6 | 5,000 additional | Resume the same parent. Use its matching full-horizon best offline WCE with fixed weights. |
| Online WCE | 6 | 5,000 additional | Resume the same parent and matching offline WCE. Update both policies. |

The first three continuations become eligible as soon as their parent finishes. Fixed WCE and Online WCE also require the matching offline WCE to finish. Each continuation starts with the full-horizon best parent’s learned controller weights and a fresh 5,000-episode training budget. Optimizer slots, update counters, learning-rate and exploration schedules, and rollout or replay storage start at their baseline initialization. DQL copies the restored online weights into its target network and begins with empty replay. Its exploration probability decreases linearly from 100% to 1% over 500,000 new control steps. MA2C and PPO sample from their restored policies and use their configured constant learning rates and entropy coefficients. A paused continuation resumes its own complete training state. Fixed WCE uses the matched offline policy for inference; Online WCE retains the matched offline WCE learner state.

## Validation and policy selection

Controllers are validated at initialization, every 100 episodes and at the horizon. Each validation replays one fixed 6,600-second demand sequence per network, comprising eleven consecutive 600-second mixtures. Offline WCE validation uses the same duration and a fixed random realization while the WCE chooses the mixtures.

For the parent and offline WCE, the first eligible run of ten validations without meaningful improvement records an early-stop result. Eligibility begins at 1,000 parent episodes or 500 WCE episodes. An improvement must reach the larger of 1% of the anchor score and 0.1 queue units in the favorable direction to reset the counter. Every strictly better raw score still qualifies as a best checkpoint. Training proceeds through the full horizon on the same trajectory.

| Result | Meaning |
|---|---|
| Early-stop best | Best observed through the first trigger, including the trigger validation. The result records both the trigger episode and the selected episode. A null early-stop record means the criterion never triggered. |
| Full-horizon best | Best validation policy from episode zero through the last episode. This policy supplies the parent or WCE for downstream training. |
| Final | Policy at the last episode, with its own validation score. |

All five continuations run through episode 5,000. Their validation history identifies the full-horizon best, and their last validation identifies the final policy. Recovery checkpoints preserve progress and the first early-stop selection across a pause. [Protocol](config/protocol.json) · [Selection rules](experiments/selection.py)

## Final evaluation and reporting

Each of the 30 continuations supplies two policy selections, **best** and **final**. Each selection runs on 11 seen and 12 generalization scenarios. Each scenario has 10 paired traffic realizations of 3,600 simulated seconds each. This yields **1,380 scenario jobs and 13,800 rollout records**. Both labels appear in reporting even when they select the same checkpoint and share cached measurements.

Mean queue is the primary outcome. Auxiliary outcomes include peak and 95th-percentile queue, completion rate, speed, and waiting and travel time for completed trips. The reports compare methods on matching network, controller, training seed, selection, scenario and traffic realization. With one formal training seed, paired rollout intervals describe traffic variability conditional on the trained policies. [Evaluation](experiments/evaluation.py) · [Reports](experiments/reporting.py)

## Review and manual release

Review the six INIs, selection evidence and protocol above before releasing the campaign. The local experiment-02 dashboard runs on loopback port 8766. While paused, the dashboard can display the archived development tasks and their actual episode counts. When the operator is ready to start, **Run automatically** authorizes verification of the frozen code and runtime, followed by formal training, both evaluation selections and reporting. A fresh passing verification gate is required for the frozen source.

For a source review, run the unit suite from the repository root with the project Python interpreter:

```bash
python3 -m unittest discover -s tests
```

The repository includes the simulator assets, experiment code, configuration and protocol evidence. Local run directories hold episode summaries, block summaries, validation records, checkpoints, scheduler receipts and evaluation shards. Detailed historical design notes in the docs directory describe earlier revisions. This README and the full_horizon_v2 protocol specify the experiment submitted for review.
